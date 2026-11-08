"""Stateless per-passage stylometric features (no fitting -- same function for any split).

Sentence/word stats from notebooks/eda.ipynb section 6, plus the punctuation fingerprints
from section 5 (curly quotes, em-dashes) that showed strong per-model separation there.

The next 5 (after the first 12) were added after a targeted ANOVA scan (not generic stylometry)
against the two confusion clusters identified in analyze_errors.py's per-class recall breakdown:
the claude<->glm bidirectional confusion, and the gpt-5.6 / gemini-3.x sibling triplets. Of a ~17
candidate battery, these 5 had the highest F-stats specifically within those hard groups
(semicolon_rate was already here and was the single strongest, F=14.22 p~1e-15 for claude-vs-glm
-- these add on top of it, since it may be getting diluted sitting among many other features).
No single one cleanly separates claude from glm as two blocks (there's comparable variation
*within* the claude family itself), so treat this as multivariate signal for the downstream
classifier, not a hand-interpretable rule.

negation_rate (17th feature) came from a direct data scan rather than the ANOVA-on-hard-clusters
process above -- it separates by PROVIDER FAMILY (all four Gemini variants cluster low, ~0.43-
0.54 negations/100 words, vs ~1.0-1.3 for every other family), not a specific confusion pair.

Everything from char_count onward (user's hand-crafted-feature list, items 12-20) is UNVALIDATED
-- added because it's cheap and plausible, same as negation_rate originally was, but none of it
has been run through a CV check yet. Treat these as candidates for train_fusion_lgbm.py's next
run to confirm or reject, not confirmed signal.
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

# -- items 12-20: hand-crafted features, untried before this --
MARKDOWN_HEADER = re.compile(r"^#{1,6}\s", re.MULTILINE)
BULLET = re.compile(r"^\s*[-*•]\s", re.MULTILINE)
NUMBERED_LIST = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
BOLD = re.compile(r"\*\*[^*\n]+\*\*")
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
HR = re.compile(r"^\s*([-*_])\1{2,}\s*$", re.MULTILINE)
ELLIPSIS = re.compile(r"\.\.\.|…")
# Common emoji Unicode blocks (emoticons, symbols/pictographs, transport, flags, supplemental
# symbols) -- regex ranges instead of the `emoji` package, since that's not a current dependency
# and this project's convention (stylometric.py, punctuation features) is regex-only so far.
EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002700-\U000027BF]"
)
LATEX_DOLLAR = re.compile(r"\$[^$\n]+\$")
LATEX_PAREN = re.compile(r"\\\([^)]+\\\)")
CODE_FENCE_LANG = re.compile(r"```[a-zA-Z]+\b")
OPENER_PHRASES = (
    "great question", "certainly", "sure", "absolutely", "i'd be happy", "i would be happy",
    "let's", "happy to help", "of course", "that's a great",
)
CLOSER_PHRASES = (
    "let me know if", "feel free to", "i hope this helps", "would you like", "don't hesitate",
    "happy to help further", "if you have any", "let me know", "hope that helps",
)
SELF_ID = re.compile(
    r"\bas an ai\b|\bi'?m an ai\b|\bi am an ai\b|\blanguage model\b|\bmy knowledge cutoff\b|"
    r"\bas a large language model\b|\bi'?m claude\b|\bi am claude\b|\bi'?m gpt\b|\bi am gpt\b|"
    r"\bi'?m gemini\b|\bi am gemini\b|\btrained by\b",
    re.IGNORECASE,
)
REFUSAL_HEDGE = re.compile(
    r"\bi cannot\b|\bi can't\b|\bi'?m unable to\b|\bi am unable to\b|"
    r"\bi don'?t have access to\b|\bit'?s worth noting\b|\bit is worth noting\b|"
    r"\bi should mention\b|\bit'?s important to note\b|\bmight\b|\bcould potentially\b|"
    r"\bperhaps\b",
    re.IGNORECASE,
)

FEATURE_NAMES = [
    "n_sentences", "mean_sent_len", "mean_word_len", "lexical_diversity",
    "comma_rate", "semicolon_rate", "exclaim_rate", "question_rate",
    "has_curly_quote", "em_dash_count", "has_2plus_em_dash",
    "sent_len_std", "colon_rate", "contraction_rate", "cap_word_rate", "this_it_the_start_rate",
    "negation_rate",
    # items 12-20, added together, untried before this:
    "char_count", "n_paragraphs",
    "markdown_header_count", "bullet_count", "numbered_list_count", "bold_count",
    "table_row_count", "hr_count",
    "hyphen_rate", "ellipsis_rate",
    "emoji_count",
    "latex_dollar_count", "latex_paren_count", "code_fence_count", "code_fence_lang_count",
    "opener_phrase_hit", "closer_phrase_hit",
    "self_id_mention_count",
    "refusal_hedge_count",
    "trailing_space_line_rate", "blank_line_rate", "line_len_std",
]


def stylometric_features(text):
    words = WORD.findall(text)
    nw = max(len(words), 1)
    sents = [s for s in SENT.split(text.strip()) if s]
    sent_lens = [len(WORD.findall(s)) for s in sents] or [0]
    word_lens = [len(w) for w in words] or [0]
    em_dash_count = text.count("—")
    n_sents = max(len(sents), 1)

    lines = text.split("\n")
    n_lines = max(len(lines), 1)
    line_lens = [len(ln) for ln in lines]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    text_lower = text.lower()
    head = text_lower[:60]
    tail = text_lower[-80:]

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
        # items 12-20
        float(len(text)),
        float(max(len(paragraphs), 1)),
        float(len(MARKDOWN_HEADER.findall(text))),
        float(len(BULLET.findall(text))),
        float(len(NUMBERED_LIST.findall(text))),
        float(len(BOLD.findall(text))),
        float(len(TABLE_ROW.findall(text))),
        float(len(HR.findall(text))),
        text.count("-") / nw,
        len(ELLIPSIS.findall(text)) / nw,
        float(len(EMOJI.findall(text))),
        float(len(LATEX_DOLLAR.findall(text))),
        float(len(LATEX_PAREN.findall(text))),
        float(text.count("```") // 2),
        float(len(CODE_FENCE_LANG.findall(text))),
        float(any(head.startswith(p) or p in head for p in OPENER_PHRASES)),
        float(any(p in tail for p in CLOSER_PHRASES)),
        float(len(SELF_ID.findall(text))),
        len(REFUSAL_HEDGE.findall(text)) / nw,
        sum(1 for ln in lines if ln.endswith((" ", "\t"))) / n_lines,
        sum(1 for ln in lines if not ln.strip()) / n_lines,
        float(np.std(line_lens)) if len(lines) > 1 else 0.0,
    ]


def stylometric_matrix(texts):
    return np.array([stylometric_features(t) for t in texts], dtype=np.float32)
