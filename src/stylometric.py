"""Stateless per-passage stylometric features (no fitting -- same function for any split).

Sentence/word stats from notebooks/eda.ipynb section 6, plus the punctuation fingerprints
from section 5 (curly quotes, em-dashes) that showed strong per-model separation there.
"""
import re

import numpy as np

SENT = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[A-Za-z’']+")

FEATURE_NAMES = [
    "n_sentences", "mean_sent_len", "mean_word_len", "lexical_diversity",
    "comma_rate", "semicolon_rate", "exclaim_rate", "question_rate",
    "has_curly_quote", "em_dash_count", "has_2plus_em_dash",
]


def stylometric_features(text):
    words = WORD.findall(text)
    nw = max(len(words), 1)
    sents = [s for s in SENT.split(text.strip()) if s]
    sent_lens = [len(WORD.findall(s)) for s in sents] or [0]
    word_lens = [len(w) for w in words] or [0]
    em_dash_count = text.count("—")
    return [
        len(sents),
        float(np.mean(sent_lens)),
        float(np.mean(word_lens)),
        len(set(w.lower() for w in words)) / nw,
        text.count(",") / nw,
        text.count(";") / nw,
        text.count("!") / nw,
        text.count("?") / nw,
        float(any(c in text for c in ("’", "“", "”"))),
        float(em_dash_count),
        float(em_dash_count >= 2),
    ]


def stylometric_matrix(texts):
    return np.array([stylometric_features(t) for t in texts], dtype=np.float32)
