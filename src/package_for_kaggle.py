"""Packages a train_fusion_lgbm.py run into kaggle_package/ for upload as a Kaggle Dataset:
flattens each fold's inference-only files (model weights, not optimizer/scheduler state) plus
the shared tokenizer, fusion artifacts, and labels, and writes dataset-metadata.json.

Usage:
    python src/package_for_kaggle.py --run_dir experiments/<run_id> \\
        --slug name-that-llm-fusion-v1 --title "name-that-llm fusion (DeBERTa+SVD-TFIDF+stylo -> LGBM)"

Then review kaggle_package/ and run (not done automatically -- this creates a dataset on your
Kaggle account):
    kaggle datasets create -p kaggle_package --dir-mode skip
"""
import argparse
import glob
import json
import os
import shutil

CHECKPOINT_KEEP = {"config.json", "model.safetensors"}  # inference-only; skip optimizer/scheduler/rng state


def package_fold(artifacts_dir, out_dir, fold):
    ckpt_hits = sorted(glob.glob(os.path.join(artifacts_dir, f"fold{fold}_ckpt", "checkpoint-*")))
    assert ckpt_hits, f"no checkpoint for fold {fold} under {artifacts_dir}"
    src_ckpt = ckpt_hits[-1]
    dst_ckpt = os.path.join(out_dir, f"fold{fold}_ckpt")
    os.makedirs(dst_ckpt, exist_ok=True)
    for name in CHECKPOINT_KEEP:
        src = os.path.join(src_ckpt, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_ckpt, name))

    fusion_src = os.path.join(artifacts_dir, f"fold{fold}_fusion.joblib")
    shutil.copy2(fusion_src, os.path.join(out_dir, f"fold{fold}_fusion.joblib"))


def resolve_run_dir(run_dir, name_glob="*fusion_lgbm*"):
    """--run_dir latest -> the most recently modified experiments/<name_glob> folder."""
    if run_dir != "latest":
        return run_dir
    from exp_log import EXPERIMENTS_ROOT
    hits = sorted(glob.glob(str(EXPERIMENTS_ROOT / name_glob)), key=os.path.getmtime)
    assert hits, f"no run directories matching {EXPERIMENTS_ROOT / name_glob}"
    return hits[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="experiments/<run_id>, or 'latest' for the most recent fusion_lgbm run")
    ap.add_argument("--out_dir", default="kaggle_package")
    ap.add_argument("--slug", required=True, help="Kaggle dataset slug (lowercase, hyphens)")
    ap.add_argument("--title", required=True)
    ap.add_argument("--username", default="ptran1203")
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    print(f"packaging from: {run_dir}")
    artifacts_dir = os.path.join(run_dir, "artifacts")
    assert os.path.isdir(artifacts_dir), f"{artifacts_dir} not found"

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    for fold in range(args.n_folds):
        package_fold(artifacts_dir, args.out_dir, fold)
        print(f"packaged fold {fold}")

    shutil.copytree(os.path.join(artifacts_dir, "tokenizer"), os.path.join(args.out_dir, "tokenizer"))
    shutil.copy2(os.path.join(artifacts_dir, "labels.json"), os.path.join(args.out_dir, "labels.json"))

    metadata = {
        "title": args.title,
        "id": f"{args.username}/{args.slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    json.dump(metadata, open(os.path.join(args.out_dir, "dataset-metadata.json"), "w"), indent=2)

    total_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(args.out_dir) for f in fs
    )
    print(f"\npackaged -> {args.out_dir} ({total_bytes / 1e9:.2f} GB)")
    print(f"dataset id will be: {metadata['id']}")
    print(f"review contents, then run:  kaggle datasets create -p {args.out_dir} --dir-mode skip")


if __name__ == "__main__":
    main()
