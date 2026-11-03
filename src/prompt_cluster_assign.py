"""Post-processing: detect which passages likely share a hidden prompt (via TF-IDF similarity,
since no prompt_id is given at inference time), then resolve within-cluster label conflicts by
rank instead of taking each row's independent argmax.

Why this can help at all: analyze_errors.py's chart of train passages sharing a prompt_id shows
responses to the same prompt converge on near-identical concrete imagery/vocabulary (same nouns,
same phrases) -- far more similar to each other than to other passages from the SAME model on a
different prompt. That makes "same prompt" detectable via plain TF-IDF cosine similarity. And
within a genuinely shared prompt, train's structure says each of the 18 models answers it at
most once, so two cluster members both getting the classifier's top vote for the same class is
suspicious -- at least one of them is wrong, and the trailing candidate class for the loser is
probably not. Greedily reassigning by confidence rank exploits that.

Calibration (see TRIED.md for the full sweep): thresholds were tuned using train's real
prompt_id as ground truth, which the test set does NOT give us. At threshold ~0.24, detected
clusters have ~0.997-1.0 precision (an edge almost never links two truly-different-prompt rows)
but partial recall (~0.6-0.8) -- i.e. this clusters CONSERVATIVELY: it misses some real
same-prompt groups (those fall back to plain argmax, no harm) rather than risking false merges
(which WOULD hurt, by forcing an incorrect exclusivity constraint on unrelated passages).

Important caveat found during calibration: the realistic gain depends heavily on how strong the
base classifier already is. On a weak TF-IDF+LR baseline the gain was large (+0.046 balanced
accuracy, detected clusters vs plain argmax). On the actual best model (fusion_lgbm, stronger
per-row predictions already), the gain shrinks to roughly +0.004-0.005 at the best threshold and
is sign-unstable (can go slightly NEGATIVE) at other nearby thresholds -- this is a small, fragile
edge on our real submission model, not the dramatic oracle-upper-bound win (+0.107, using TRUE
prompt_id groups) this is modeled on. Ship it as a low-risk, modest-upside option, not a
guaranteed win.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import normalize

from train import build_vectorizers, fit_transform

DEFAULT_THRESHOLD = 0.24


def cosine_sim_matrix(texts):
    """Dense pairwise cosine similarity over TF-IDF (same word+char vectorizer as the main
    pipeline), fit fresh on exactly the given texts -- unsupervised structure detection, no
    labels involved, so this is safe to fit directly on the test set itself."""
    word_vec, char_vec, _punct_vec, _func_vec = build_vectorizers()
    X = fit_transform(texts, word_vec, char_vec, fit=True).tocsr()
    Xn = normalize(X, norm="l2")
    sim = (Xn @ Xn.T).toarray()
    np.fill_diagonal(sim, 0.0)
    return sim


def detect_clusters(texts, threshold=DEFAULT_THRESHOLD):
    """Returns an int array (len(texts),) of connected-component cluster ids. Rows with no
    edge above `threshold` end up as their own singleton cluster -- callers should treat
    singleton clusters as "no information", i.e. leave them on plain argmax."""
    sim = cosine_sim_matrix(texts)
    adj = sim > threshold
    np.fill_diagonal(adj, False)
    _, labels = connected_components(csr_matrix(adj), directed=False)
    return labels


def resolve_cluster_assignments(proba, cluster_labels, max_cluster_size=18):
    """Greedy rank-based conflict resolution within each detected cluster (IDEAS.md #5 /
    TRIED.md): sort every (row, class) candidate pair in the cluster by probability descending,
    assign greedily, skip a pair if its row or its class is already taken. Clusters of size 1
    (no detected match) or > max_cluster_size (likely a false merge -- no real cluster should
    exceed the number of classes) are left on plain argmax, untouched.

    proba: [n_rows, n_classes] probability matrix (NaN rows, if any, are skipped -- same as
    plain argmax would also fail on them). Returns an int array of predicted class indices."""
    pred = np.argmax(np.nan_to_num(proba, nan=-1.0), axis=-1).copy()
    valid = ~np.isnan(proba).any(axis=1)

    by_cluster = {}
    for i, c in enumerate(cluster_labels):
        if valid[i]:
            by_cluster.setdefault(c, []).append(i)

    for members in by_cluster.values():
        n = len(members)
        if n < 2 or n > max_cluster_size:
            continue
        sub = proba[members]
        triples = sorted(
            ((sub[r, k], r, k) for r in range(n) for k in range(sub.shape[1])),
            reverse=True,
        )
        used_rows, used_cols, assign = set(), set(), {}
        for _, r, k in triples:
            if r in used_rows or k in used_cols:
                continue
            assign[r] = k
            used_rows.add(r)
            used_cols.add(k)
            if len(used_rows) == n:
                break
        for r, k in assign.items():
            pred[members[r]] = k

    return pred
