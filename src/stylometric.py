"""Stateless per-passage stylometric features (no fitting -- same function for any split).

Sentence/word stats from notebooks/eda.ipynb section 6, plus the punctuation fingerprints
from section 5 (curly quotes, em-dashes) that showed strong per-model separation there.

The last 5 features were added after a targeted ANOVA scan (not generic stylometry) against
the two confusion clusters identified in analyze_errors.py's per-class recall breakdown: the
claude<->glm bidirectional confusion, and the gpt-5.6 / gemini-3.x sibling triplets. Of a ~17
candidate battery, these 5 had the highest F-stats specifically within those hard groups
(semicolon_rate was already here and was the single strongest, F=14.22 p~1e-15 for claude-vs-glm
-- these add on top of it, since it may be getting diluted sitting among many other features).
No single one cleanly separates claude from glm as two blocks (there's comparable variation
*within* the claude family itself), so treat this as multivariate signal for the downstream
classifier, not a hand-interpretable rule.

negation_rate (17th feature) came from a direct data scan rather than the ANOVA-on-hard-clusters
process above -- it separates by PROVIDER FAMILY (all four Gemini variants cluster low, ~0.43-
0.54 negations/100 words, vs ~1.0-1.3 for every other family), not a specific confusion pair.
Not yet validated end-to-end through train_fusion_lgbm.py (that requires a fine-tuned transformer
checkpoint for the embedding side) -- added because the raw separation is unusually large and
clean, but treat it as unconfirmed until that CV check actually runs.
"""
import re

import numpy as np

SENT = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[A-Za-z’']+")
CONTRACTION = re.compile(r"\w+['’](t|s|re|ve|ll|d|m)\b")
THIS_IT_THE_START = re.compile(r"^(This|It|The)\b")
# Negation rate: found via a direct data scan (not a generic guess) -- cleanly separates by
# PROVIDER FAMILY rather than resolving a specific known confusion pair. All four Gemini variants
# cluster low (~0.43-0.54 negations per 100 words) while every other family sits much higher
# (~1.0-1.3), a ~3x spread, bigger than most other features here.
NEGATION = re.compile(r"\b(not|never|cannot|nothing|none|neither|nor)\b|n['’]t\b", re.IGNORECASE)

FEATURE_NAMES = [
    "n_sentences", "mean_sent_len", "mean_word_len", "lexical_diversity",
    "comma_rate", "semicolon_rate", "exclaim_rate", "question_rate",
    "has_curly_quote", "em_dash_count", "has_2plus_em_dash",
    "sent_len_std", "colon_rate", "contraction_rate", "cap_word_rate", "this_it_the_start_rate",
    "negation_rate",
]


def stylometric_features(text):
    words = WORD.findall(text)
    nw = max(len(words), 1)
    sents = [s for s in SENT.split(text.strip()) if s]
    sent_lens = [len(WORD.findall(s)) for s in sents] or [0]
    word_lens = [len(w) for w in words] or [0]
    em_dash_count = text.count("—")
    n_sents = max(len(sents), 1)
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
        float(np.std(sent_lens)) if len(sents) > 1 else 0.0,
        text.count(":") / nw,
        len(CONTRACTION.findall(text)) / nw,
        sum(1 for w in words if w[:1].isupper()) / nw,
        sum(1 for s in sents if THIS_IT_THE_START.match(s.strip())) / n_sents,
        len(NEGATION.findall(text)) / nw,
    ]


def stylometric_matrix(texts):
    return np.array([stylometric_features(t) for t in texts], dtype=np.float32)
