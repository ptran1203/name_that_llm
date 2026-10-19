"""Resolves data/model paths across local, Kaggle, and Colab environments."""
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

_DATA_DIR_CANDIDATES = [
    pathlib.Path("/kaggle/input/competitions/name-that-llm"),
    pathlib.Path("/kaggle/input/name-that-llm"),
    _REPO_ROOT / "data",
    pathlib.Path("data"),
]


def data_dir():
    for p in _DATA_DIR_CANDIDATES:
        if (p / "train.csv").exists():
            return p
    raise FileNotFoundError(
        "train.csv not found. Looked in: " + ", ".join(str(p) for p in _DATA_DIR_CANDIDATES)
    )


def models_dir():
    # /kaggle/working is the only writable dir in a Kaggle notebook; everything else is read-only.
    p = pathlib.Path("/kaggle/working/models") if pathlib.Path("/kaggle/working").exists() else _REPO_ROOT / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p
