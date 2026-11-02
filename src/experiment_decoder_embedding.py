"""Frozen decoder (causal) LM embeddings -> plain classifier.

Alternative embedding source to the encoder (DeBERTa-v3) mean-pooled embeddings already tested
in notebooks/frozen_embeddings.ipynb. Encoder models use bidirectional attention, so every
token has already seen the whole sequence and mean-pooling makes sense. Causal/decoder models
only attend backward, so only the LAST token has seen the entire passage -- earlier tokens are
under-informed. Pool via that last real token's hidden state instead (same approach as
decoder-based embedding models like E5-Mistral/LLM2Vec/GritLM), using left-padding so the last
real token is reliably at index -1 for every row in a batch, no per-row index gathering needed.

Embeddings are extracted ONCE (frozen, no fine-tuning -- depends only on text, not on fold
assignment), then GroupKFold-by-prompt_id CV just fits a plain classifier on them, same
leakage discipline as train.py.
"""
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_log import ExperimentRun
from paths import data_dir

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_decoder_embeddings(model_name, texts, max_length=256, batch_size=16, log=print,
                                pooling="last_token"):
    """pooling='last_token' (default): theoretically correct for causal attention -- only the
    final real token has seen the whole sequence. pooling='mean': averages every token's hidden
    state anyway (including under-informed early tokens) -- included only to check whether
    last-token pooling specifically is the bottleneck, or the decoder model's representations are
    weak regardless of how they're pooled."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:  # most decoder-only models have no pad token by default
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # keeps the last REAL token at index -1 for every row

    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])
            enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True,
                             return_tensors="pt").to(DEVICE)
            hidden = model(**enc, output_hidden_states=True).hidden_states[-1]
            if pooling == "mean":
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                pooled = hidden[:, -1, :]
            out.append(pooled.float().cpu().numpy())
            if (i // batch_size) % 20 == 0:
                log(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


def make_clf_factory(name):
    # Unlike DeBERTa's bounded mean-pooled embeddings, raw causal-LM hidden states aren't
    # scale-normalized -- LogisticRegression failed to converge without StandardScaler first.
    if name == "svm":
        # LinearSVC has no predict_proba -- decision_function still works fine for argmax-based
        # accuracy, just isn't a calibrated probability (fine here, not fine if stacking later).
        return lambda: make_pipeline(StandardScaler(), LinearSVC(max_iter=5000, class_weight="balanced"))
    return lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced", n_jobs=-1)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B",
                     help="any causal/decoder HF model, e.g. Qwen/Qwen2.5-0.5B, "
                          "HuggingFaceTB/SmolLM2-360M, meta-llama/Llama-3.2-1B")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--classifier", default="lr", choices=["lr", "svm"])
    ap.add_argument("--pooling", default="last_token", choices=["last_token", "mean"],
                     help="last_token (default, theoretically correct for causal attention) or "
                          "mean (included only to check whether last_token pooling itself is the "
                          "bottleneck, or the decoder model's representations are weak regardless)")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--max_folds", type=int, default=None,
                     help="stop after this many folds, for a quick single-fold check")
    ap.add_argument("--cache_dir", default="data/embed_cache",
                     help="cache extracted embeddings here, keyed by model_name+max_length+"
                          "pooling, so re-running with a different --classifier skips extraction")
    args = ap.parse_args()

    config = vars(args)
    run_name = f"decoder_embed_{args.model_name.split('/')[-1]}_{args.pooling}_{args.classifier}"
    run = ExperimentRun(run_name, config=config)

    df = pd.read_csv(data_dir() / "train.csv")
    class_labels = sorted(df["model"].unique())
    label2id = {l: i for i, l in enumerate(class_labels)}

    import hashlib
    from pathlib import Path
    cache_key = hashlib.md5(f"{args.model_name}_{args.max_length}_{args.pooling}".encode()).hexdigest()[:10]
    cache_path = Path(args.cache_dir) / f"{cache_key}.npy"
    if cache_path.exists():
        run.log(f"loading cached embeddings -> {cache_path}")
        emb = np.load(cache_path)
    else:
        run.log(f"extracting {args.pooling}-pooled embeddings from {args.model_name} (device={DEVICE})...")
        emb = extract_decoder_embeddings(
            args.model_name, df["text"].tolist(), max_length=args.max_length,
            pooling=args.pooling,
            batch_size=args.batch_size, log=run.log,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
        run.log(f"cached embeddings -> {cache_path}")
    run.log(f"embedding shape: {emb.shape}")

    make_clf = make_clf_factory(args.classifier)
    gkf = GroupKFold(n_splits=args.n_splits)
    oof_proba = np.full((len(df), len(class_labels)), np.nan, dtype=np.float32)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df["model"], groups=df["prompt_id"])):
        if args.max_folds is not None and fold >= args.max_folds:
            break
        clf = make_clf()
        y_train = df["model"].values[train_idx]
        clf.fit(emb[train_idx], y_train)

        fold_score = np.zeros((len(val_idx), len(class_labels)), dtype=np.float32)
        if hasattr(clf, "predict_proba"):
            fold_score[:, [label2id[c] for c in clf.classes_]] = clf.predict_proba(emb[val_idx])
        else:
            fold_score[:, [label2id[c] for c in clf.classes_]] = clf.decision_function(emb[val_idx])
        oof_proba[val_idx] = fold_score

        pred = [class_labels[i] for i in np.argmax(fold_score, axis=-1)]
        score = balanced_accuracy_score(df["model"].values[val_idx], pred)
        scores.append(score)
        run.log(f"fold {fold}: balanced_accuracy={score:.4f}")

    run.log_scores("balanced_accuracy", scores)
    np.save(run.artifact_path("oof_proba.npy"), oof_proba)
    run.log("saved OOF predictions -> oof_proba.npy")


if __name__ == "__main__":
    main()
