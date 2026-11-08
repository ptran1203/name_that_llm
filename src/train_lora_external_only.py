"""LoRA fine-tune a decoder model on EXTERNAL data ONLY (no official train.csv rows), then save
the adapter + tokenizer + pooling metadata for embed_lora_cv.py to extract embeddings from.

Why external-only: the frozen-embedding concat experiment showed that concatenating external
rows into a classifier already anchored on official data doesn't help (TRIED.md, flat-to-
negative). This script tests the other end of the spectrum -- train the backbone itself purely
on the large external pool, then see (via embed_lora_cv.py) whether the resulting embedding
transfers to the official distribution better than one trained on official data alone.

Reuses build_model/LayerPoolClassifier/TextDataset/DEVICE from train_qwen_lora.py so both
scripts share one LoRA/pooling implementation. No GroupKFold here -- external data has no
prompt_id grouping relationship to the competition's official prompts, so there's nothing to
fold over; this trains one model on (capped) external data and saves it.

Usage (Kaggle): pip install -q peft bitsandbytes catboost  (if not already present), then:
    python src/train_lora_external_only.py \
        --external_paths /kaggle/input/name-that-llm-open-pool/ntl_open_pool.parquet \
        --external_cap_per_class 2000 --output_dir lora_external_only
"""
import argparse

import pandas as pd
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer

from exp_log import ExperimentRun
from experiment_multi_embed_concat import load_external_data
from paths import data_dir
from train_qwen_lora import TextDataset, build_model
from train_transformer import build_training_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--external_paths", required=True,
                     help="comma-separated parquet path(s) of external data, e.g. the "
                          "name-that-llm-open-pool dataset's ntl_open_pool.parquet")
    ap.add_argument("--external_cap_per_class", type=int, default=2000,
                     help="rows per class to sample from the external pool -- the dataset "
                          "author's own ablation used 2500/class (45k total) for a 1.7B model; "
                          "2000 here is a comparable order of magnitude")
    ap.add_argument("--pooling_layer", type=int, default=7,
                     help="hidden_states index to pool from -- default is the empirically-best "
                          "frozen layer for Qwen2.5-3B (TRIED.md), NOT the last layer")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--no_gradient_checkpointing", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true",
                     help="QLoRA -- needed once the base model doesn't fit alongside LoRA's "
                          "activation memory otherwise (e.g. 7B+ on a 16GB GPU)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="lora_external_only",
                     help="where to save the LoRA adapter, tokenizer, and pooling metadata")
    args = ap.parse_args()

    if torch.cuda.device_count() > 1:
        print("WARNING: >1 GPU visible. transformers.Trainer silently wraps in DataParallel in "
              "that case, which has caused multiple confirmed crash bugs on this project "
              "(TRIED.md). Set CUDA_VISIBLE_DEVICES=0 before running this script.")

    run = ExperimentRun(f"lora_external_only_{args.model_name.split('/')[-1]}", config=vars(args))

    # label2id must match the official competition labels, even though no official rows are
    # trained on here -- the saved classifier head's label order (and embed_lora_cv.py's
    # downstream labels) must line up with train.csv's.
    official_df = pd.read_csv(data_dir() / "train.csv")
    class_labels = sorted(official_df["model"].unique())
    label2id = {l: i for i, l in enumerate(class_labels)}

    ext_df = load_external_data(args.external_paths, args.external_cap_per_class, label2id, run.log)
    ext_df["label"] = ext_df["model"].map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = TextDataset(ext_df["text"], ext_df["label"], tokenizer, args.max_length)

    model = build_model(args.model_name, len(class_labels), args.pooling_layer,
                         args.lora_r, args.lora_alpha, args.lora_dropout,
                         not args.no_gradient_checkpointing, args.precision,
                         args.load_in_4bit, run.log)

    training_args = build_training_args(dict(
        output_dir=str(run.artifact_path("ckpt")),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),
        seed=args.seed,
        report_to=[],
    ))
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    run.log(f"training on {len(ext_df)} external rows ({args.external_cap_per_class}/class cap)...")
    trainer.train()

    # Save only the LoRA adapter (small -- base weights are frozen and not re-saved) + tokenizer
    # + the pooling metadata embed_lora_cv.py needs to reconstruct the same model and extract
    # embeddings from the same layer. The classifier head trained here is NOT saved -- it was
    # only needed to produce a gradient signal during training; embed_lora_cv.py reads
    # `model.pool()` output directly, never the head.
    model.base_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    import json
    with open(f"{args.output_dir}/pooling_metadata.json", "w") as f:
        json.dump({
            "model_name": args.model_name,
            "pooling_layer": args.pooling_layer,
            "max_length": args.max_length,
        }, f, indent=2)
    run.log(f"saved LoRA adapter + tokenizer + pooling metadata -> {args.output_dir}")


if __name__ == "__main__":
    main()
