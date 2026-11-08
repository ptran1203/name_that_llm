"""Use a LoRA adapter trained by train_lora_external_only.py to extract embeddings for ALL of
train.csv (official data), then 5-fold GroupKFold CV those embeddings into several classical
ML models (lgbm, lr, svc, catboost) -- same leakage discipline (group by prompt_id) as every
other CV in this project.

This is the "what did external-only fine-tuning actually learn" check: unlike the frozen-
embedding concat experiment (train_qwen_lora.py's base model, never fine-tuned, TRIED.md), this
embedding comes from a backbone that was fine-tuned purely on external data -- so if the
representation transferred, this CV score should beat the un-fine-tuned frozen-embedding
baseline for the same base model/layer (0.3978 for Qwen2.5-3B layer 7, full 5-fold).

Usage (Kaggle), after running train_lora_external_only.py:
    pip install -q peft bitsandbytes catboost
    python src/embed_lora_cv.py --adapter_dir lora_external_only
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from transformers import AutoModel, AutoTokenizer

from exp_log import ExperimentRun
from paths import data_dir

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_lora_model(adapter_dir, load_in_4bit, log):
    meta = json.loads((Path(adapter_dir) / "pooling_metadata.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        base = AutoModel.from_pretrained(meta["model_name"], quantization_config=bnb_config,
                                          device_map={"": DEVICE})
    else:
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        base = AutoModel.from_pretrained(meta["model_name"], torch_dtype=dtype).to(DEVICE)

    model = PeftModel.from_pretrained(base, adapter_dir).eval()
    log(f"loaded LoRA adapter from {adapter_dir} (base={meta['model_name']}, "
        f"layer={meta['pooling_layer']})")
    return model, tokenizer, meta


def extract_embeddings(model, tokenizer, texts, layer, max_length, batch_size, log):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])
            enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True,
                             return_tensors="pt").to(DEVICE)
            hidden = model(**enc, output_hidden_states=True).hidden_states[layer]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.float().cpu().numpy())
            if (i // batch_size) % 20 == 0:
                log(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return np.concatenate(out, axis=0)


def make_clf(name):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, verbosity=-1)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, verbose=False)
    if name == "svc":
        from sklearn.svm import SVC
        return make_pipeline(StandardScaler(), SVC(class_weight="balanced"))
    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, C=5.0, class_weight="balanced", n_jobs=-1))
    raise ValueError(f"unknown classifier {name!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True,
                     help="output_dir from train_lora_external_only.py")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--load_in_4bit", action="store_true",
                     help="must match whether train_lora_external_only.py used it -- the "
                          "adapter was trained against either a 4-bit or full-precision base")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--max_folds", type=int, default=None,
                     help="stop after this many folds, for a quick single-fold check")
    ap.add_argument("--classifiers", default="lgbm,lr,svc,catboost")
    ap.add_argument("--cache_path", default=None,
                     help="where to cache extracted embeddings (.npy) -- default: "
                          "<adapter_dir>/train_embeddings.npy")
    args = ap.parse_args()

    run = ExperimentRun("embed_lora_cv", config=vars(args))
    df = pd.read_csv(data_dir() / "train.csv")
    class_labels = sorted(df["model"].unique())
    label2id = {l: i for i, l in enumerate(class_labels)}
    df["label"] = df["model"].map(label2id)

    cache_path = Path(args.cache_path or f"{args.adapter_dir}/train_embeddings.npy")
    if cache_path.exists():
        run.log(f"loading cached embeddings -> {cache_path}")
        embeddings = np.load(cache_path)
    else:
        model, tokenizer, meta = load_lora_model(args.adapter_dir, args.load_in_4bit, run.log)
        embeddings = extract_embeddings(model, tokenizer, df["text"].tolist(), meta["pooling_layer"],
                                         meta["max_length"], args.batch_size, run.log)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        run.log(f"saved embeddings {embeddings.shape} -> {cache_path}")

    gkf = GroupKFold(n_splits=args.n_splits)
    folds = list(gkf.split(df, df["model"], groups=df["prompt_id"]))
    if args.max_folds is not None:
        folds = folds[:args.max_folds]

    for clf_name in args.classifiers.split(","):
        scores = []
        try:
            for fold, (train_idx, val_idx) in enumerate(folds):
                clf = make_clf(clf_name)
                clf.fit(embeddings[train_idx], df["label"].values[train_idx])
                pred = clf.predict(embeddings[val_idx])
                score = balanced_accuracy_score(df["label"].values[val_idx], pred)
                scores.append(score)
                run.log(f"[{clf_name}] fold {fold}: balanced_accuracy={score:.4f}")
        except Exception as e:
            # One classifier's dependency (e.g. catboost) failing to import/fit in a given
            # environment shouldn't block results from the others.
            run.log(f"[{clf_name}] FAILED, skipping: {e}")
            continue
        run.log_scores(f"balanced_accuracy_{clf_name}", scores)


if __name__ == "__main__":
    main()
