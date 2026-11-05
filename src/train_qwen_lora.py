"""LoRA fine-tune a frozen decoder model (Qwen2.5-3B by default) for 18-way classification.

Why not AutoModelForSequenceClassification: its default head pools the LAST layer via the last
real token -- found to be the WORST frozen layer for this task (TRIED.md's layer sweep: 0.2893
at the last layer vs 0.3978 at layer 7). LoRA's low-rank updates are deliberately small, so
fine-tuning alone isn't trusted to fully relocate that signal to the last layer in a few epochs
on ~2700 rows. Instead this wraps the base model directly, pools from a CONFIGURABLE layer
(default 7, the empirically-best frozen layer) via mean-pooling over the attention mask -- same
pooling already validated in experiment_decoder_embedding.py -- then a fresh linear head.

LoRA (peft): freezes the base model, inserts small trainable low-rank adapters into attention
projections. Only the adapters + the new classifier head get gradients/optimizer state --
full fine-tuning a 3B model's optimizer state alone wouldn't fit a single consumer/Kaggle GPU.
Gradients still flow back through every frozen layer up to the pooling layer, so activation
memory is controlled via gradient checkpointing + small batch + gradient accumulation instead.

GroupKFold by prompt_id, single GPU, same leakage discipline as train_transformer.py. Test
--max_folds 1 first before committing to a full run -- this is new, heavier infrastructure.

Usage (Kaggle): pip install -q peft  (if not already present in the image), then:
    python src/train_qwen_lora.py --max_folds 1 --batch_size 4 --grad_accum_steps 4
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, Trainer

from exp_log import ExperimentRun
from paths import data_dir
from train_transformer import build_training_args

# Under accelerate/DDP each process has its own assigned GPU -- PartialState().device resolves
# to the right one for THIS process regardless of launch method (accelerate launch, torchrun,
# plain python), unlike hand-parsing LOCAL_RANK ourselves. A bare "cuda" always means cuda:0,
# which would put every process's model on the same GPU and break DDP.
DEVICE = PartialState().device


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item


class LayerPoolClassifier(nn.Module):
    """base_model (frozen + LoRA) -> hidden_states[layer] -> mean-pool over attention mask ->
    linear head -> logits. The classifier head is left in float32 (standard practice for
    numerical stability even when the backbone runs in bf16/fp16) -- pooled activations are
    cast to match before the final matmul."""

    def __init__(self, base_model, hidden_size, num_labels, layer):
        super().__init__()
        self.base_model = base_model
        self.layer = layer
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                               output_hidden_states=True)
        hidden = out.hidden_states[self.layer]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        logits = self.classifier(pooled.to(self.classifier.weight.dtype))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


def build_model(model_name, num_labels, layer, lora_r, lora_alpha, lora_dropout,
                 gradient_checkpointing, precision, load_in_4bit, log):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

    if load_in_4bit:
        # QLoRA: the frozen base stays in 4-bit (~1/4 the memory of bf16), LoRA adapters still
        # train in `dtype`. Needed once the base model itself is too big to fit alongside LoRA's
        # activation memory otherwise (e.g. 7B+ on a single 16GB GPU) -- not needed at 3B.
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
        )
        # device_map must point at THIS process's own assigned GPU (DEVICE, derived from
        # LOCAL_RANK) -- a hardcoded device would put every DDP process's model on the same GPU.
        base = AutoModel.from_pretrained(model_name, quantization_config=bnb_config,
                                          device_map={"": str(DEVICE)})
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=gradient_checkpointing)
    else:
        base = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(DEVICE)
    hidden_size = base.config.hidden_size

    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="FEATURE_EXTRACTION",
    )
    base = get_peft_model(base, lora_config)
    if gradient_checkpointing and not load_in_4bit:  # prepare_model_for_kbit_training already enabled it
        base.gradient_checkpointing_enable()
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    total = sum(p.numel() for p in base.parameters())
    log(f"LoRA base: {trainable:,}/{total:,} trainable ({100 * trainable / total:.3f}%)")

    model = LayerPoolClassifier(base, hidden_size, num_labels, layer).to(DEVICE)
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"balanced_accuracy": balanced_accuracy_score(labels, preds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--pooling_layer", type=int, default=7,
                     help="hidden_states index to pool from -- default is the empirically-best "
                          "frozen layer for Qwen2.5-3B (TRIED.md), NOT the last layer")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4,
                     help="per-device batch size -- kept small since activation memory for a "
                          "3B model's full forward/backward graph is the real VRAM constraint "
                          "under LoRA, not optimizer state")
    ap.add_argument("--grad_accum_steps", type=int, default=4,
                     help="effective batch size = batch_size * grad_accum_steps")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--max_folds", type=int, default=None,
                     help="stop after this many folds -- test with 1 before a full run, this "
                          "is new infrastructure")
    ap.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"],
                     help="bf16 needs no loss scaler (unlike fp16) -- safer given this project's "
                          "documented fp16 GradScaler crash history. Use fp32 on a GPU without "
                          "bf16 support (e.g. T4/P100).")
    ap.add_argument("--no_gradient_checkpointing", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true",
                     help="QLoRA: quantize the frozen base to 4-bit (bitsandbytes) before "
                          "attaching LoRA adapters -- needed once the base model itself doesn't "
                          "fit alongside LoRA's activation memory (e.g. 7B+ on a 16GB GPU). Not "
                          "needed at 3B. Requires `pip install bitsandbytes`.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if torch.cuda.device_count() > 1:
        print("WARNING: >1 GPU visible. transformers.Trainer silently wraps in DataParallel in "
              "that case, which has caused multiple confirmed crash bugs on this project "
              "(TRIED.md). Set CUDA_VISIBLE_DEVICES=0 before running this script.")

    run = ExperimentRun(f"qwen_lora_{args.model_name.split('/')[-1]}", config=vars(args))
    df = pd.read_csv(data_dir() / "train.csv")
    class_labels = sorted(df["model"].unique())
    label2id = {l: i for i, l in enumerate(class_labels)}
    df["label"] = df["model"].map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gkf = GroupKFold(n_splits=args.n_splits)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df["model"], groups=df["prompt_id"])):
        if args.max_folds is not None and fold >= args.max_folds:
            break
        run.log(f"=== fold {fold} ===")
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        train_ds = TextDataset(train_df["text"], train_df["label"], tokenizer, args.max_length)
        val_ds = TextDataset(val_df["text"], val_df["label"], tokenizer, args.max_length)

        model = build_model(args.model_name, len(class_labels), args.pooling_layer,
                             args.lora_r, args.lora_alpha, args.lora_dropout,
                             not args.no_gradient_checkpointing, args.precision,
                             args.load_in_4bit, run.log)

        training_args = build_training_args(dict(
            output_dir=str(run.artifact_path(f"fold{fold}_ckpt")),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            gradient_accumulation_steps=args.grad_accum_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            bf16=(args.precision == "bf16"),
            fp16=(args.precision == "fp16"),
            seed=args.seed,
            report_to=[],
        ))

        trainer = Trainer(
            model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
            data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics,
        )
        trainer.train()
        eval_result = trainer.evaluate()
        score = eval_result["eval_balanced_accuracy"]
        scores.append(score)
        run.log(f"fold {fold}: balanced_accuracy={score:.4f}")

        del model, trainer
        torch.cuda.empty_cache()

    run.log_scores("balanced_accuracy", scores)


if __name__ == "__main__":
    main()
