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
                                pooling="last_token", layers=(-1,)):
    """pooling='last_token': theoretically correct for causal attention -- only the final real
    token has seen the whole sequence. pooling='mean' (default here): averages every token's
    hidden state anyway -- found better than last_token in an earlier small-model test (IDEAS.md
    reference notebook research), consistent with public solutions using mean-pooled hidden
    states rather than last-token.

    layers: which hidden_states indices to extract (0 = embedding output, 1..num_layers = each
    transformer block's output, negative counts from the end). Extracts ALL requested layers in
    ONE forward pass per batch -- a 3B+ model is too expensive to reload per layer. Public
    top-scoring solutions for this competition use MIDDLE layers (e.g. Qwen3-8B L7+L22, not the
    last layer) rather than assuming the final layer is best -- this sweeps several to check.
    Returns {layer_idx: np.array[n, hidden]}."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:  # most decoder-only models have no pad token by default
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # keeps the last REAL token at index -1 for every row

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE).eval()
    out = {layer: [] for layer in layers}
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])
            enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True,
                             return_tensors="pt").to(DEVICE)
            all_hidden = model(**enc, output_hidden_states=True).hidden_states
            mask = enc["attention_mask"].unsqueeze(-1).float()
            for layer in layers:
                hidden = all_hidden[layer]
                if pooling == "mean":
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                else:
                    pooled = hidden[:, -1, :]
                out[layer].append(pooled.float().cpu().numpy())
            if (i // batch_size) % 20 == 0:
                log(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    n_layers = len(all_hidden)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    log(f"  model has {n_layers} hidden_states entries (embedding + transformer blocks)")
    return {layer: np.concatenate(v, axis=0) for layer, v in out.items()}


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
    ap.add_argument("--pooling", default="mean", choices=["last_token", "mean"],
                     help="mean (default -- found better than last_token in an earlier small-"
                          "model test) or last_token (theoretically correct for causal attention, "
                          "but empirically worse here)")
    ap.add_argument("--layers", default="-1",
                     help="comma-separated hidden_states indices to sweep in one model load, "
                          "e.g. '-1,19,13,7' (negative = from the end; 0 = embedding output). "
                          "Public top solutions for this competition use MIDDLE layers, not the "
                          "last one -- this runs a full CV per layer so they're comparable.")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--max_folds", type=int, default=None,
                     help="stop after this many folds, for a quick single-fold check")
    ap.add_argument("--cache_dir", default="data/embed_cache",
                     help="cache extracted embeddings here, keyed by model_name+max_length+"
                          "pooling+layer, so re-running with a different --classifier skips extraction")
    args = ap.parse_args()
    layers = tuple(int(x) for x in args.layers.split(","))

    config = vars(args)
    df = pd.read_csv(data_dir() / "train.csv")
    class_labels = sorted(df["model"].unique())
    label2id = {l: i for i, l in enumerate(class_labels)}

    import hashlib
    from pathlib import Path
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path_for(layer):
        key = hashlib.md5(f"{args.model_name}_{args.max_length}_{args.pooling}_{layer}".encode()).hexdigest()[:10]
        return cache_dir / f"{key}.npy"

    missing = [l for l in layers if not cache_path_for(l).exists()]
    emb_by_layer = {l: np.load(cache_path_for(l)) for l in layers if l not in missing}
    if missing:
        print(f"extracting {args.pooling}-pooled embeddings from {args.model_name} for layers "
              f"{missing} (device={DEVICE})...")
        fresh = extract_decoder_embeddings(
            args.model_name, df["text"].tolist(), max_length=args.max_length,
            pooling=args.pooling, layers=missing, batch_size=args.batch_size, log=print,
        )
        for l, arr in fresh.items():
            np.save(cache_path_for(l), arr)
            emb_by_layer[l] = arr

    make_clf = make_clf_factory(args.classifier)
    gkf = GroupKFold(n_splits=args.n_splits)
    for layer in layers:
        emb = emb_by_layer[layer]
        run = ExperimentRun(
            f"decoder_embed_{args.model_name.split('/')[-1]}_L{layer}_{args.pooling}_{args.classifier}",
            config={**config, "layer": layer, "embedding_shape": list(emb.shape)},
        )
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
            run.log(f"layer {layer} fold {fold}: balanced_accuracy={score:.4f}")

        run.log_scores("balanced_accuracy", scores)
        np.save(run.artifact_path("oof_proba.npy"), oof_proba)


if __name__ == "__main__":
    main()
