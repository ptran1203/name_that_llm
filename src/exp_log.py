"""Common experiment logging: experiments/<run_id>/{run_info.json, scores.json, run.log, artifacts/}.

Usage:
    run = ExperimentRun("tfidf_baseline", config={"C": 5.0, "ngram_range": [1, 2]})
    run.log("starting CV...")
    run.log_scores("balanced_accuracy", fold_scores)
    joblib.dump(model, run.artifact_path("model.joblib"))

Later, compare all runs with `load_experiments()` (returns a pandas DataFrame) or
`python src/compare_experiments.py`.
"""
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent / "experiments"


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=EXPERIMENTS_ROOT.parent, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


class ExperimentRun:
    """One row in experiments/: run_info.json (config/metadata), scores.json (metrics), run.log, artifacts/."""

    def __init__(self, name, config=None, root=EXPERIMENTS_ROOT):
        self.name = name
        self.run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"
        self.dir = Path(root) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self._write_run_info()
        self.logger = self._make_logger()
        self.log(f"run_id={self.run_id}")
        if self.config:
            self.log(f"config={json.dumps(self.config)}")

    def _write_run_info(self):
        info = {
            "run_id": self.run_id,
            "name": self.name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": _git_commit(),
            "config": self.config,
        }
        (self.dir / "run_info.json").write_text(json.dumps(info, indent=2))

    def _make_logger(self):
        logger = logging.getLogger(self.run_id)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fh = logging.FileHandler(self.dir / "run.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.propagate = False
        return logger

    def log(self, msg):
        self.logger.info(msg)

    def log_scores(self, metric_name, fold_scores, extra=None):
        """Records one named metric's per-fold scores + mean/std. Safe to call multiple times
        with different metric_name values in the same run (e.g. 'plain' vs 'mixup')."""
        import numpy as np

        scores_path = self.dir / "scores.json"
        all_scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
        entry = {
            "fold_scores": [float(s) for s in fold_scores],
            "mean": float(np.mean(fold_scores)),
            "std": float(np.std(fold_scores)),
        }
        if extra:
            entry.update(extra)
        all_scores[metric_name] = entry
        scores_path.write_text(json.dumps(all_scores, indent=2))
        self.log(f"{metric_name}: mean={entry['mean']:.4f} +/- {entry['std']:.4f} (folds={entry['fold_scores']})")
        return entry

    def artifact_path(self, filename):
        p = self.dir / "artifacts"
        p.mkdir(exist_ok=True)
        return p / filename

    def log_curve(self, tag, records):
        """For deep-learning runs: records is a list of per-epoch dicts, e.g.
        [{"epoch": 1, "train_loss": 0.9, "eval_loss": 2.1, "eval_balanced_accuracy": 0.2}, ...].
        Saves '<tag>_curve.json' and a matplotlib plot '<tag>_curve.png' under artifacts/."""
        records = sorted(records, key=lambda r: r["epoch"])
        json_path = self.artifact_path(f"{tag}_curve.json")
        json_path.write_text(json.dumps(records, indent=2))
        png_path = self._plot_curve(tag, records)
        self.log(f"logged training curve '{tag}' ({len(records)} epochs) -> {json_path.name}, {png_path.name}")
        return json_path

    def _plot_curve(self, tag, records):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [r["epoch"] for r in records]
        loss_keys = [k for k in ("train_loss", "eval_loss") if any(k in r for r in records)]
        metric_keys = [k for k in records[0] if k not in ("epoch", "train_loss", "eval_loss")]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        for k in loss_keys:
            ax1.plot(epochs, [r.get(k) for r in records], marker="o", label=k)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax1.legend(loc="upper left")

        if metric_keys:
            ax2 = ax1.twinx()
            for k in metric_keys:
                ax2.plot(epochs, [r.get(k) for r in records], marker="s", linestyle="--", color="green", label=k)
            ax2.set_ylabel(", ".join(metric_keys))
            ax2.legend(loc="upper right")

        fig.suptitle(f"{self.name} - {tag}")
        fig.tight_layout()
        png_path = self.artifact_path(f"{tag}_curve.png")
        fig.savefig(png_path)
        plt.close(fig)
        return png_path


def load_experiments(root=EXPERIMENTS_ROOT):
    """One row per run, metric columns flattened as '<metric>.mean' / '<metric>.std'."""
    import pandas as pd

    root = Path(root)
    if not root.exists():
        return pd.DataFrame()

    rows = []
    for run_dir in sorted(root.iterdir()):
        info_path = run_dir / "run_info.json"
        if not info_path.exists():
            continue
        info = json.loads(info_path.read_text())
        row = {
            "run_id": info["run_id"],
            "name": info["name"],
            "timestamp": info["timestamp"],
            "git_commit": info.get("git_commit"),
        }
        for k, v in info.get("config", {}).items():
            row[f"cfg.{k}"] = v

        scores_path = run_dir / "scores.json"
        if scores_path.exists():
            scores = json.loads(scores_path.read_text())
            for metric_name, entry in scores.items():
                row[f"{metric_name}.mean"] = entry.get("mean")
                row[f"{metric_name}.std"] = entry.get("std")
        rows.append(row)

    df = pd.DataFrame(rows)
    mean_cols = [c for c in df.columns if c.endswith(".mean")]
    if mean_cols:
        df["best_mean"] = df[mean_cols].max(axis=1)
        df = df.sort_values("best_mean", ascending=False)
    return df
