"""Frozen (no fine-tuning) layer-wise embeddings -> plain classifier, for EITHER a decoder
(causal) model or an encoder (bidirectional) model.

Why sweep layers at all: the LAST layer isn't automatically the best one for a transfer/probing
task like this -- it's whatever the pretraining objective favors, which for a causal LM is
next-token prediction specifically (very task-specific, found empirically worse here: 0.29 at
the last layer vs 0.40 at an early-middle layer for Qwen2.5-3B, non-monotonic). For an encoder,
every layer already has full bidirectional context, but the SAME "last layer overfits to the
pretraining objective" argument could still apply (e.g. masked-token prediction specifics) --
untested before this script, since notebooks/frozen_embeddings.ipynb only ever pooled the final
layer. This script checks both architectures with the same harness.

Pooling: mean over all real (non-padding) tokens, masked correctly regardless of pad side --
found better than last-token pooling for causal models (an earlier small-model test), and is
the standard choice for encoders regardless (every token already has full context either way).

Embeddings are extracted ONCE per (model, layer) (frozen -- depends only on text, not on fold
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
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from exp_log import ExperimentRun
from paths import data_dir

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_decoder_embeddings(model_name, texts, max_length=256, batch_size=16, log=print,
                                pooling="mean", layers=(-1,), architecture="causal",
                                load_in_4bit=False):
    """architecture='causal' (AutoModelForCausalLM, e.g. Qwen) or 'encoder' (AutoModel, e.g.
    DeBERTa/RoBERTa/ELECTRA) -- same layer-sweep + pooling logic either way, since mean-pooling
    with the attention mask is correct regardless of padding side or attention direction.

    layers: which hidden_states indices to extract (0 = embedding output, 1..num_layers = each
    transformer block's output, negative counts from the end). Extracts ALL requested layers in
    ONE forward pass per batch -- a multi-GB model is too expensive to reload per layer. Public
    top-scoring solutions for this competition use MIDDLE layers (e.g. Qwen3-8B L7+L22, not the
    last layer) rather than assuming the final layer is best -- this sweeps several to check.
    Returns {layer_idx: np.array[n, hidden]}."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:  # most decoder-only models have no pad token by default
        tokenizer.pad_token = tokenizer.eos_token
    if architecture == "causal":
        tokenizer.padding_side = "left"  # irrelevant for mean pooling, kept for last_token pooling

    model_cls = AutoModelForCausalLM if architecture == "causal" else AutoModel
    if load_in_4bit:
        # Needed once a model's fp16 weights alone don't fit the GPU (e.g. Qwen2.5-7B, ~15GB,
        # on a 12GB card) -- same BitsAndBytesConfig as train_qwen_lora.py's QLoRA path, just
        # for frozen extraction (no adapters, no training) instead.
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        model = model_cls.from_pretrained(
            model_name, quantization_config=bnb_config, device_map={"": DEVICE},
        ).eval()
    else:
        model = model_cls.from_pretrained(
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
                elif pooling == "cls":
                    pooled = hidden[:, 0, :]  # first token -- BERT-style [CLS]/<s>
                else:
                    pooled = hidden[:, -1, :]  # last_token
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
                          "HuggingFaceTB/SmolLM2-360M, meta-llama/Llama-3.2-1B -- or, with "
                          "--architecture encoder, microsoft/deberta-v3-base, roberta-base, etc.")
    ap.add_argument("--architecture", default="causal", choices=["causal", "encoder"],
                     help="causal (AutoModelForCausalLM, e.g. Qwen) or encoder (AutoModel, e.g. "
                          "DeBERTa/RoBERTa/ELECTRA) -- controls model class and whether "
                          "last_token pooling would even make sense (encoders: always mean).")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--classifier", default="lr", choices=["lr", "svm"])
    ap.add_argument("--pooling", default="mean", choices=["last_token", "mean", "cls"],
                     help="mean (default), last_token (causal models, empirically worse than "
                          "mean), or cls (first-token pooling, e.g. BERT-style [CLS]/<s> -- the "
                          "token BERT-family pretraining actually optimizes for sentence-level "
                          "representation, untested here before now)")
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
    ap.add_argument("--load_in_4bit", action="store_true",
                     help="load the model in 4-bit (bitsandbytes) for extraction -- needed once "
                          "fp16 weights alone don't fit the GPU (e.g. Qwen2.5-7B on a 12GB card). "
                          "Requires `pip install bitsandbytes`.")
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
        # suffix only when non-default, so --load_in_4bit=False (the default, matching every
        # previously cached run) keeps hitting the same cache keys as before this flag existed.
        suffix = "_4bit" if args.load_in_4bit else ""
        key = hashlib.md5(
            f"{args.model_name}_{args.architecture}_{args.max_length}_{args.pooling}_{layer}{suffix}".encode()
        ).hexdigest()[:10]
        return cache_dir / f"{key}.npy"

    missing = [l for l in layers if not cache_path_for(l).exists()]
    emb_by_layer = {l: np.load(cache_path_for(l)) for l in layers if l not in missing}
    if missing:
        print(f"extracting {args.pooling}-pooled embeddings from {args.model_name} "
              f"({args.architecture}) for layers {missing} (device={DEVICE})...")
        fresh = extract_decoder_embeddings(
            args.model_name, df["text"].tolist(), max_length=args.max_length,
            pooling=args.pooling, layers=missing, architecture=args.architecture,
            batch_size=args.batch_size, log=print, load_in_4bit=args.load_in_4bit,
        )
        for l, arr in fresh.items():
            np.save(cache_path_for(l), arr)
            emb_by_layer[l] = arr

    make_clf = make_clf_factory(args.classifier)
    gkf = GroupKFold(n_splits=args.n_splits)
    for layer in layers:
        emb = emb_by_layer[layer]
        run = ExperimentRun(
            f"{args.architecture}_embed_{args.model_name.split('/')[-1]}_L{layer}_{args.pooling}_{args.classifier}",
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
