# Prior competition research

Findings from researching structurally-similar prior text-classification competitions, looking
specifically for techniques we haven't tried or might be doing wrong. Searched via web search
(no direct Kaggle access), so some details are secondhand (blog posts / GitHub repos summarizing
discussion threads rather than the primary source) -- flagged where that's the case.

## Kaggle "LLM - Detect AI Generated Text" (2023-2024)

The single most directly relevant competition: binary human-vs-AI text classification, heavy
focus on TF-IDF + classical-ML ensembles beating pure transformer fine-tunes for much of the
competition.

- **Custom BPE tokenizer trained from scratch on the competition corpus, fed into TF-IDF.** This
  was the signature winning trick: train a byte-level BPE tokenizer (HuggingFace `tokenizers`,
  NFC normalization) on the combined train+test text itself (unsupervised -- no labels needed,
  so using test text for vocab-building isn't leakage), then run standard TF-IDF vectorization
  over *those* subword tokens instead of word-level tokens.
  [github.com/baseline repo](https://github.com/toshan07/LLM-AI-Generated-Text-Detection),
  [topic-level writeups](https://github.com/topics/wordpiece)
- **Heavy ensembling of diverse classical models on top of the TF-IDF features**: MultinomialNB,
  SGD, CatBoost, LightGBM, RandomForest, weighted-blended (not just one model).
  [Lizhecheng02 repo, ~0.96-0.97 LB](https://github.com/Lizhecheng02/Kaggle-LLM-Detect_AI_Generated_Text)
- **1st place solution fine-tuned LLMs (QLoRA) to *generate more synthetic training essays***
  mimicking the target (PERSUADE) corpus style, to grow effective training data, plus a DeBERTa
  ranking-model component.
  [nhyha/llm-detect-ai (1st place)](https://github.com/nhyha/llm-detect-ai),
  [discussion #473295 (comprehensive 1st place write-up, not opened directly)](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/473295)

## Kaggle "Spooky Author Identification" (2017-2018)

Structurally the closest analog: 3-class authorship attribution (Poe/Lovecraft/Shelley) from
short text snippets (single sentences, even shorter than our 40-250 word passages), scored by
multi-class log loss (not balanced accuracy, but still a per-class-sensitive metric since classes
are balanced).

- **Progression that won**: bag-of-words -> TF-IDF + NB-SVM-style Logistic Regression + Complement
  Naive Bayes -> adding punctuation/character-level features (improved both accuracy and macro-F1
  over word-only) -> a stacked ensemble on out-of-fold predictions as the final step.
  [Towards Data Science: "How Far Can Classical NLP Go?"](https://towardsdatascience.com/how-far-can-classical-nlp-go-from-bag-of-words-to-stacking-on-spooky-author-identification/)
  (secondhand via search excerpts, full article not opened)
- A simple linear SVM/LR on word features alone was reported as the strongest single model by at
  least one independent writeup, ahead of CNN/LSTM neural approaches on this amount of data.
  [harkash/spooky_author_identification](https://github.com/harkash/spooky_author_identification)

## Jigsaw Toxic Comment Classification Challenge (2018)

Binary/multi-label (6 toxicity types) comment classification, not authorship, but the two
specific techniques below are general and directly applicable regardless of task.

- **NBSVM (Naive-Bayes-weighted SVM/Logistic Regression)**: scale TF-IDF features by Naive Bayes
  log-count ratios before feeding a linear classifier, instead of using raw/plain TF-IDF weights.
  Originally Wang & Manning 2012, "Baselines and Bigrams" (ACL), popularized for this exact
  competition via Jeremy Howard's "NB-SVM strong linear baseline" Kaggle kernel.
  [Wang & Manning 2012 paper](https://preview.aclanthology.org/fix_video/P12-2018.pdf),
  [reimplementation + results](https://github.com/jeanvc/nbsvm)
- **Diverse pretrained embeddings mattered more than later-layer architecture** per a secondhand
  summary of the 1st place writeup (0.9890 AUC) -- the embedding layer was called out as the most
  important component.
  [NLP Approaches on Kaggle slides, citing discussion #52557](https://www.slideshare.net/slideshow/kaggle-nlp-approaches/129822729)
- **Pseudo-labeling + test-time augmentation** were both credited with leaderboard gains in a
  secondhand summary (0.9877 -> 0.9880 -> 0.9885).
  [Toxic Comment Classification Challenge summary](https://nycdatascience.com/blog/student-works/toxic-comment-classification-challenge-a-kaggle-competition)
  (these exact figures didn't fully reconcile against the 0.9890 above across sources -- treat as
  directionally right, not exact)

## Jigsaw Multilingual Toxic Comment Classification (2020)

- **1st place**: XLM-RoBERTa ensembles + pseudo-labeling, per Google's Perspective API paper.
  [PaLM 2 Technical Report, citing the result](https://arxiv.org/pdf/2305.10403) (indirect
  citation, not the primary writeup)
- **Pseudo-labeling external/translated data, not just the test set**: one team's pipeline
  trained XLM-R on the English-only official train set, used it to predict pseudo-labels for
  translations of those same English comments into other languages, and separately pseudo-labeled
  the test set itself.
  [Team Lingua Franca notes](https://wandb.ai/wandb_fc/articles/reports/Kaggle-Jigsaw-Multilingual-Toxic-Comment-Classification--Vmlldzo1NDI3MDc3)
- **3rd place (Team ACE, 0.9523)**: blend of four model families, plus pseudo-labeled Open
  Subtitles data as additional training signal.
  [3rd place solution writeup](https://towardsdatascience.com/kaggle-3rd-place-solution-jigsaw-multilingual-toxic-comment-classification-e36d7d194bfb)

## PAN / CLEF authorship attribution shared tasks (academic, not Kaggle)

Multiple years (2013, 2018, 2019) of cross-domain authorship attribution, closest academic analog
to our actual task (attributing a *specific* author/source, not just human-vs-AI).

- **Consistent finding across years: character n-grams are the single most informative feature
  type**, ahead of word n-grams and POS tags, typically in an SVM.
  [Bartelds & de Vries, PAN 2019](https://research.rug.nl/en/publications/dbcb6426-6bae-418b-8bdf-67b92a389a2b)
  (macro-F1 0.644, char n-grams called out as most informative)
- **2nd place PAN 2019**: ensemble of separate SVMs per feature type (char, word, stemmed-word,
  distorted-text n-grams), ~68% F1 on dev data.
  [PAN 2019 2nd place, Rome team](https://iris.uniroma1.it/handle/11573/1446340)
- **Punctuation n-grams as their own dedicated feature channel** (not just punctuation rates/
  counts, but n-grams *of* punctuation sequences, preserving order) added measurable signal on
  top of standard character n-grams.
  [punctuation n-grams + averaged character n-grams, PAN-affiliated](https://ipn.elsevierpure.com/es/publications/authorship-attribution-through-punctuation-n-grams-and-averaged-c)

## Other (lower priority, general technique confirmation only)

- **Quora Insincere Questions Classification**: search didn't surface a clean 1st-place writeup
  (one widely-discussed incident: a team hit 1st on the public leaderboard by scraping answers
  from quora.com directly -- a leaderboard-probing exploit, not a generalizable technique). The
  closest adjacent competition with a documented pseudo-labeling pipeline was Google QUEST Q&A
  Labeling (1st place: fine-tune on StackExchange data, then pseudo-label).
  [Quora silver-medal (182/4037) writeup](https://towardsdatascience.com/what-my-first-silver-medal-taught-me-about-text-classification-and-kaggle-in-general-ebae0df16278)

---

## Relevance to our problem

Mapped against our actual gaps (claude<->glm confusion, sibling-model confusion, small per-class
N, incomplete stacking/external-data results, balanced-accuracy optimization). Each item marked
**NEW** (not in IDEAS.md/TRIED.md), **REINFORCES** (external validation of something already in
flight), or **NOT APPLICABLE** (doesn't transfer to our setup, with why).

1. **NEW -- NBSVM (Naive Bayes-weighted TF-IDF features).** We currently feed plain/raw TF-IDF
   weights into LogisticRegression (`train.py`). NBSVM scales each feature by its Naive Bayes
   log-count ratio first -- cheap (one extra step before `.fit()`), well-documented, and was a
   genuine competition-winning technique on text classification specifically. Untried here.

2. **NEW -- train our own BPE/subword tokenizer from scratch on our corpus, vs. reusing a
   pretrained one.** We just confirmed reusing DeBERTa-v3's *pretrained* subword tokenizer for
   TF-IDF gives +0.03 balanced accuracy (0.3382 vs 0.3086). The Detect-AI-Generated-Text winners
   did something related but distinct: train a BPE tokenizer *from scratch* on the competition's
   own text (train+test combined, unsupervised). That's corpus-specific vocabulary instead of
   general-purpose pretrained vocabulary -- could capture vocabulary/subword patterns specific to
   *our* 18 models' outputs rather than whatever DeBERTa happened to learn from its own
   pretraining corpus. Genuinely different lever from what we just tried, cheap to test
   (HuggingFace `tokenizers` library, `BPE` + a `WordLevelTrainer`-style loop over `df['text']`).

3. **NEW -- punctuation n-grams as a dedicated feature channel (order-preserving), not just
   punctuation rates.** Our stylometric features count punctuation *rates*
   (`comma_rate`, `semicolon_rate`, etc. -- see `src/stylometric.py`), which loses sequence/order
   information. PAN's punctuation n-gram approach extracts just the punctuation stream from each
   text (preserving order) and runs character n-gram TF-IDF over *that* -- distinct from both our
   rate features (no order) and our general char n-grams (punctuation would be drowned out in a
   3-5 char window dominated by letters). Cheap: regex-extract punctuation-only string per
   passage, run a small char-n-gram TfidfVectorizer over it, concatenate into the existing fusion.
   Directly relevant to the claude<->glm confusion given punctuation (curly quotes, semicolons)
   is already our strongest known signal there.

4. **REINFORCES -- stacking (OOF meta-classifier) and heavy ensembling of diverse models.** Both
   the Detect-AI-Generated-Text winners (NB+SGD+CatBoost+LightGBM+RandomForest blend) and the
   Spooky Author Identification writeup (stacking as the final/best step) converge on the same
   thing already identified as our most promising incomplete result in TRIED.md (stacking dropped
   the train/val gap from ~0.59 to ~0.10-0.13, never finished to a full 5-fold mean). External
   validation that this is the right next priority, not new information.

5. **REINFORCES -- character n-grams as the strongest single stylometric feature type.** PAN's
   consistent finding across 2013/2018/2019 validates our existing char n-gram (3-5) TF-IDF
   component (`train.py`) and the semicolon-rate/punctuation-heavy stylometric features already
   added. Not a new lever, but confirms we're not missing an obviously-better feature family.

6. **PARTIALLY NEW -- pseudo-labeling *external unlabeled data*, not the competition test set.**
   We already discussed and deprioritized pseudo-labeling our own ~4000-row test set directly
   (risk of reinforcing existing blind spots on hard classes, given current balanced accuracy
   ~0.32-0.35 isn't high-confidence enough). Jigsaw Multilingual's winners did something more
   specific: pseudo-label a *separate* unlabeled corpus (translations of labeled data) with a
   trained model, then add high-confidence pseudo-labels as extra weighted training data -- same
   discipline we already apply to the external data sources (train-fold-only, down-weighted), just
   generating the labels ourselves instead of relying on a third party's labeling. Could apply to
   `ntl_open_pool.parquet` rows that don't have confident existing labels, or any other unlabeled
   LLM-text source we can find. Not in IDEAS.md currently.

7. **NOT APPLICABLE -- generating more training data by directly querying the source LLMs.** The
   Detect-AI-Generated-Text 1st place solution fine-tuned LLMs to generate more synthetic essays
   mimicking the target corpus. Doesn't transfer here: our 18 labels are fictional/future-versioned
   model names (`claude-fable-5`, `gpt-6-astra`, etc.) that aren't independently queryable --
   there's no real API to go generate "more claude-fable-5 text" from. Worth noting explicitly so
   this doesn't get re-proposed later without re-deriving why it's a dead end.

8. **Balanced-accuracy / per-class-averaged metric optimization**: none of the researched
   competitions used balanced accuracy specifically (Jigsaw used AUC, Detect-AI-Generated-Text
   used AUC-ROC, Spooky Author used multi-class log loss). PAN's macro-F1 is the closest analog
   (also per-class-averaged, also insensitive to class-size imbalance). No specific optimization
   trick surfaced beyond what we already do (`class_weight="balanced"`, GroupKFold-by-prompt_id to
   avoid topic leakage) -- and we already specifically tested and rejected post-hoc predicted-
   marginal calibration as a balanced-accuracy lever (TRIED.md: made it worse, 0.3482 -> 0.3441,
   because balanced accuracy normalizes by true count per class, not predicted frequency). No new
   finding here.
