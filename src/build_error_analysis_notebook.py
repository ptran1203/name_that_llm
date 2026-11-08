"""One-off builder for notebooks/error_analysis.ipynb -- run this, don't hand-edit the notebook's
code cells (regenerate from here instead, same convention as build_frozen_embed_notebook.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md("""# Error analysis -- best single result (frozen Qwen2.5-3B layer-7 embedding + LR)

No new training here. This reconstructs out-of-fold predictions for the best result found so
far (`decoder_embed_Qwen2.5-3B_L7_mean_lr`, full 5-fold GroupKFold CV, balanced_accuracy=0.3978
-- TRIED.md's layer-depth sweep) from its cached `oof_proba.npy`, then looks at *where* the
errors are: confusion matrix, per-class recall, same-family vs cross-family confusions,
confidence calibration, and concrete misclassified examples.

0.3978 on 18 classes is ~7.2x the random floor (1/18=0.056) -- real signal, but we're still
wrong on the majority of rows. The question this notebook tries to answer: is that because the
*features* are weak, or because many of these 18 classes are genuinely hard to tell apart (e.g.
same-vendor adjacent versions like `claude-opus-4-6`/`4-8`/`5`, or the `gpt-5.6-luna/sol/terra`
triplet)? If errors concentrate in those known-similar clusters rather than spreading evenly,
that's a real ceiling from stylistic overlap, not a fixable modeling gap.""")

code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import sys
sys.path.insert(0, '../src')
from paths import data_dir
from exp_log import EXPERIMENTS_ROOT""")

md("""## 1. Load data and reconstruct OOF predictions

`experiment_decoder_embedding.py` loads `train.csv` with no shuffling and fills `oof_proba`
at the same row positions GroupKFold assigned as validation -- so re-loading `train.csv` the
same way lines back up with the cached `oof_proba.npy` exactly.""")

code("""RUN_DIR = EXPERIMENTS_ROOT / "20261103-223628_decoder_embed_Qwen2.5-3B_L7_mean_lr"

df = pd.read_csv(data_dir() / "train.csv")
labels_sorted = sorted(df["model"].unique())

oof_proba = np.load(RUN_DIR / "artifacts" / "oof_proba.npy")
assert not np.isnan(oof_proba).any(), "some rows never got a fold assignment"

df["true_label"] = df["model"]
df["pred_label"] = [labels_sorted[i] for i in oof_proba.argmax(axis=1)]
df["confidence"] = oof_proba.max(axis=1)
df["correct"] = df["true_label"] == df["pred_label"]
df["n_words"] = df["text"].str.split().str.len()

bal_acc = balanced_accuracy_score(df["true_label"], df["pred_label"])
plain_acc = df["correct"].mean()
print(f"balanced accuracy (should match TRIED.md's 0.3978): {bal_acc:.4f}")
print(f"plain accuracy: {plain_acc:.4f}  (close to balanced acc -> classes are ~evenly sized: "
      f"{df['true_label'].value_counts().min()}-{df['true_label'].value_counts().max()} rows/class)")""")

md("""## 2. Confusion matrix

Row-normalized (each row sums to 1) so the diagonal reads directly as per-class recall.""")

code("""cm = confusion_matrix(df["true_label"], df["pred_label"], labels=labels_sorted)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(labels_sorted)))
ax.set_yticks(range(len(labels_sorted)))
ax.set_xticklabels(labels_sorted, rotation=60, ha="right", fontsize=8)
ax.set_yticklabels(labels_sorted, fontsize=8)
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title("Confusion matrix (row-normalized = per-class recall)\\nfrozen Qwen2.5-3B L7 + LR, 0.3978 CV")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of true-class rows")
fig.tight_layout()
plt.show()""")

md("""## 3. Per-class recall

Which classes does the model actually nail, and which does it barely get above chance?""")

code("""recalls = np.diag(cm_norm)
order = np.argsort(recalls)
labels_by_recall = [labels_sorted[i] for i in order]
recalls_sorted = recalls[order]
mean_recall = recalls.mean()
random_floor = 1 / len(labels_sorted)

fig, ax = plt.subplots(figsize=(8, 7))
colors = ["#d62728" if r < mean_recall else "#2ca02c" for r in recalls_sorted]
ax.barh(range(len(labels_by_recall)), recalls_sorted, color=colors)
ax.set_yticks(range(len(labels_by_recall)))
ax.set_yticklabels(labels_by_recall, fontsize=9)
ax.set_xlabel("recall")
ax.set_title("Per-class recall")
ax.axvline(mean_recall, color="black", linestyle="--", linewidth=1, label=f"mean = {mean_recall:.3f}")
ax.axvline(random_floor, color="gray", linestyle=":", linewidth=1, label=f"random floor = {random_floor:.3f}")
for i, r in enumerate(recalls_sorted):
    ax.text(r + 0.01, i, f"{r:.2f}", va="center", fontsize=8)
ax.set_xlim(0, 1)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
plt.show()""")

md("""## 4. Same-family vs cross-family errors

`FAMILY_PREFIXES` groups by vendor (claude/gpt/gemini/glm/qwen/kimi) -- the coarse grouping.
`HARD_CLUSTERS` are specific *adjacent-version* groups flagged earlier (TRIED.md) as suspected
hardest to tell apart: these are a stricter, more specific hypothesis than "same vendor" --
e.g. `claude-fable-5` is a different product line from `claude-opus-*`/`claude-sonnet-5`, so
vendor-level grouping alone would understate how concentrated the real confusion is.""")

code("""FAMILY_PREFIXES = ["claude", "gpt", "gemini", "glm", "qwen", "kimi"]

def family_of(label):
    for p in FAMILY_PREFIXES:
        if label.startswith(p):
            return p
    return label

HARD_CLUSTERS = {
    "claude-opus (adjacent versions)": {"claude-opus-4-6", "claude-opus-4-8", "claude-opus-5"},
    "gpt-5.6 triplet": {"gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"},
    "gemini flash (adjacent versions)": {"gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"},
    "glm (adjacent variants)": {"glm-5.3", "glm-5.3-flash"},
}

def cluster_of(label):
    for name, members in HARD_CLUSTERS.items():
        if label in members:
            return name
    return None

errors = df[~df["correct"]].copy()
errors["true_family"] = errors["true_label"].map(family_of)
errors["pred_family"] = errors["pred_label"].map(family_of)
within_family = (errors["true_family"] == errors["pred_family"]).mean()
print(f"Of {len(errors)} errors: {within_family:.1%} stay within the same vendor family, "
      f"{1 - within_family:.1%} cross vendor families.")

errors["true_cluster"] = errors["true_label"].map(cluster_of)
errors["pred_cluster"] = errors["pred_label"].map(cluster_of)
in_hard_cluster = errors["true_cluster"].notna()
within_cluster = (errors.loc[in_hard_cluster, "true_cluster"] == errors.loc[in_hard_cluster, "pred_cluster"])
print(f"Of the {in_hard_cluster.sum()} errors whose true label is in one of the named hard "
      f"clusters, {within_cluster.mean():.1%} were predicted as another member of the SAME "
      f"cluster (vs {1/17:.1%} if errors were spread evenly over the other 17 classes).")

print("\\nTop 15 confusion pairs (true -> predicted):")
pairs = []
for i, a in enumerate(labels_sorted):
    for j, b in enumerate(labels_sorted):
        if i != j and cm[i, j] > 0:
            pairs.append((cm_norm[i, j], cm[i, j], a, b))
pairs.sort(reverse=True)
for rate, cnt, a, b in pairs[:15]:
    tag = "same-family" if family_of(a) == family_of(b) else "cross-family"
    hc = " [SAME HARD CLUSTER]" if cluster_of(a) is not None and cluster_of(a) == cluster_of(b) else ""
    print(f"  {a:26s} -> {b:26s}  rate={rate:.2f}  n={cnt:3d}  ({tag}){hc}")""")

md("""## 5. Confidence: correct vs incorrect predictions

If the model is well-calibrated, wrong predictions should cluster at *lower* confidence than
right ones -- i.e. the model "knows" when it's guessing. If wrong predictions are just as
confident as right ones, that's a more concerning failure mode (confidently wrong, not
calibrated uncertainty).""")

code("""fig, ax = plt.subplots(figsize=(7, 4.5))
bins = np.linspace(0, 1, 25)
ax.hist(df.loc[df["correct"], "confidence"], bins=bins, alpha=0.6, label="correct", color="#2ca02c")
ax.hist(df.loc[~df["correct"], "confidence"], bins=bins, alpha=0.6, label="incorrect", color="#d62728")
ax.set_xlabel("predicted-class probability (confidence)")
ax.set_ylabel("count")
ax.set_title("Confidence distribution: correct vs incorrect")
ax.legend()
fig.tight_layout()
plt.show()

print(df.groupby("correct")["confidence"].describe()[["mean", "50%", "min", "max"]])""")

md("""## 6. Does passage length affect accuracy?

Prompts target 40-250 words. If short passages are harder (less stylistic signal to work
with), accuracy should drop at the low end of that range.""")

code("""df["len_bucket"] = pd.cut(df["n_words"], bins=[0, 80, 120, 160, 200, 300],
                          labels=["<80", "80-120", "120-160", "160-200", "200+"])
acc_by_len = df.groupby("len_bucket")["correct"].agg(["mean", "size"])
print(acc_by_len)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(acc_by_len.index.astype(str), acc_by_len["mean"], color="#1f77b4")
ax.axhline(plain_acc, color="black", linestyle="--", linewidth=1, label=f"overall = {plain_acc:.3f}")
ax.set_xlabel("passage length (words)")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy by passage length")
ax.legend()
fig.tight_layout()
plt.show()""")

md("""## 7. Reading actual misclassifications from the worst confusion pairs

Confidence is listed for the *predicted* (wrong) label, so a high number here means the model
was fooled convincingly, not just guessing.""")

code("""for rate, cnt, true_lbl, pred_lbl in pairs[:3]:
    print(f"\\n=== true={true_lbl}  predicted={pred_lbl}  (rate={rate:.2f}, n={cnt}) ===")
    examples = df[(df["true_label"] == true_lbl) & (df["pred_label"] == pred_lbl)].sample(
        min(2, cnt), random_state=0)
    for _, row in examples.iterrows():
        snippet = row["text"][:300].replace("\\n", " ")
        print(f"  [confidence={row['confidence']:.2f}] {snippet}...")""")

md("""## Takeaways

Fill in after running: which clusters dominate the errors, whether confidence is well
calibrated, and whether length is a real confound -- these determine whether the next useful
move is better features/more data, or accepting a ceiling from genuinely similar same-vendor
writing styles.""")

nb["cells"] = cells
with open("notebooks/error_analysis.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote notebooks/error_analysis.ipynb")
