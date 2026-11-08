"""Embedding-separation losses/machinery for the "treat this like face recognition" direction
(user-proposed) -- pull same-class pooled embeddings together and/or push different-class ones
apart, instead of relying on cross-entropy alone to shape the representation.

Center loss (Wen et al. 2016, "A Discriminative Feature Learning Approach for Deep Face
Recognition") already lives in train_qwen_lora.py's LayerPoolClassifier._center_loss -- that one
only needs an EMA per-class center, no same-class pairs within a batch, so it didn't need
anything from this file. The three below all genuinely need multiple samples of the same class
co-present in one batch (positive pairs), which a batch_size=4-from-18-classes random sampler
essentially never gives you -- hence PKBatchSampler.

- `supcon_loss`: Khosla et al. 2020, "Supervised Contrastive Learning" -- every other same-class
  sample in the batch is a positive, everything else a negative (not just one pair at a time).
  Validated specifically for few-labels-per-class text classification (Sosnowski & Wroblewska
  2022, arXiv 2211.15202; Jian et al. 2022, arXiv 2205.01308) at 16 examples/class, close to our
  ~150/class regime.
- `batch_hard_triplet_loss`: Hermans et al. 2017 ("In Defense of the Triplet Loss") batch-hard
  mining -- per anchor, only the hardest (furthest) same-class positive and hardest (nearest)
  different-class negative count. WhosAI (arXiv 2407.09364) uses exactly this for LLM-generator
  attribution specifically, then a nearest-centroid classifier at inference.
- `ArcMarginProduct`: ArcFace (Deng et al. 2019, arXiv 1801.07698) additive angular margin
  softmax head. No NLP/text precedent found when researching this (every hit was face/vision) --
  included as the explicitly lower-confidence/exploratory option, not a validated-for-text
  technique like the other two.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler


def supcon_loss(features, labels, temperature=0.07):
    """features: [n, d] pooled embeddings (any dtype -- cast to float32 internally). labels:
    [n] int. Rows with no same-class partner elsewhere in the batch contribute 0 (can't form a
    positive pair) -- make sure PKBatchSampler gives k>=2 per class or every row is wasted."""
    features = F.normalize(features.float(), dim=1)
    n = features.size(0)
    sim = (features @ features.T) / temperature

    self_mask = torch.eye(n, dtype=torch.bool, device=features.device)
    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.T) & ~self_mask

    sim = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    pos_count = positive_mask.sum(dim=1)
    has_positive = pos_count > 0
    if not has_positive.any():
        return features.new_tensor(0.0)
    # masked_fill'd self-entries are -inf -- log_prob * positive_mask would hit -inf*0=nan there,
    # so zero them out via `where` instead of multiplying.
    masked_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob))
    per_anchor_loss = -masked_log_prob.sum(dim=1)[has_positive] / pos_count[has_positive]
    return per_anchor_loss.mean()


def batch_hard_triplet_loss(features, labels, margin=0.3):
    """features: [n, d], labels: [n] int. Anchors with no same-class partner AND no different-
    class row elsewhere in the batch (shouldn't happen with PKBatchSampler, p>=2) contribute 0."""
    features = features.float()
    dist = torch.cdist(features, features, p=2)
    n = features.size(0)
    labels = labels.view(-1, 1)
    eye = torch.eye(n, dtype=torch.bool, device=features.device)
    same = (labels == labels.T) & ~eye
    diff = labels != labels.T

    hardest_pos = dist.masked_fill(~same, -1.0).max(dim=1).values
    hardest_neg = dist.masked_fill(~diff, float("inf")).min(dim=1).values

    valid = same.any(dim=1) & diff.any(dim=1)
    if not valid.any():
        return features.new_tensor(0.0)
    loss = F.relu(hardest_pos - hardest_neg + margin)
    return loss[valid].mean()


class ArcMarginProduct(nn.Module):
    """ArcFace head: cosine similarity to per-class weight vectors, with an additive angular
    margin applied to the TRUE class only during training (never at eval -- pass labels=None for
    plain scaled-cosine logits, used for argmax prediction/eval metrics)."""

    def __init__(self, in_features, out_features, scale=30.0, margin=0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.scale = scale
        self.margin = margin

    def forward(self, x, labels=None):
        x = F.normalize(x.float(), dim=1)
        w = F.normalize(self.weight, dim=1)
        cosine = x @ w.T
        if labels is None:
            return cosine * self.scale
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target_logit = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).to(cosine.dtype)
        output = cosine * (1 - one_hot) + target_logit * one_hot
        return output * self.scale


class PKBatchSampler(Sampler):
    """Yields batches of P distinct classes x K samples each (batch_size = P*K), the standard
    "PK sampling" scheme metric-learning training needs to guarantee same-class positive pairs
    actually co-occur in a batch -- a plain random sampler at batch_size=4 over 18 classes
    almost never gives you two samples of the same class together.

    hard_clusters: optional list of label-id sets (e.g. the adjacent-version clusters from
    TRIED.md/error_analysis.ipynb: claude-opus-4-6/4-8/5, gpt-5.6-luna/sol/terra, gemini-flash
    3.6/3.7/3.8, glm-5.3/5.3-flash). With probability hard_cluster_prob, one full cluster is
    forced into the batch's P classes -- deliberately creating the hard negative pairs IDEAS.md
    #6 and the WhosAI/HALvest-Contrastive papers both call out as what actually matters for
    metric learning (random negatives are mostly "easy", cross-family, and teach the model
    little)."""

    def __init__(self, labels, p, k, hard_clusters=None, hard_cluster_prob=0.5, seed=0):
        self.labels = np.asarray(labels)
        self.classes = sorted(set(self.labels.tolist()))
        self.by_class = {c: np.where(self.labels == c)[0] for c in self.classes}
        self.p = min(p, len(self.classes))
        self.k = k
        self.hard_clusters = [set(c) for c in (hard_clusters or []) if len(c) <= self.p]
        self.hard_cluster_prob = hard_cluster_prob
        self.rng = np.random.RandomState(seed)
        self.n_batches = max(1, len(self.labels) // (self.p * self.k))

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            chosen = []
            if self.hard_clusters and self.rng.random() < self.hard_cluster_prob:
                cluster = list(self.hard_clusters[self.rng.randint(len(self.hard_clusters))])
                chosen.extend(cluster)
            remaining = [c for c in self.classes if c not in chosen]
            self.rng.shuffle(remaining)
            while len(chosen) < self.p and remaining:
                chosen.append(remaining.pop())
            chosen = chosen[:self.p]

            batch = []
            for c in chosen:
                pool = self.by_class[c]
                replace = len(pool) < self.k
                picked = self.rng.choice(pool, size=self.k, replace=replace)
                batch.extend(int(i) for i in picked)
            self.rng.shuffle(batch)
            yield batch
