# Methodology — Business Entity Resolution

**Result:** final leaderboard macro F0.5 = **0.983** (public 0.98275). Held-out India/US ≈ 0.988; France (unseen country) ≈ 0.95.
**Data used:** only the challenge files. No external data, lookups, geocoding or APIs.
**Models:** `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, ~118M) and `xlm-roberta-base` (MIT, ~278M), both fine-tuned on the training labels.

## 1. Problem framing
Each Source 1 (S1) record must be linked to all of its records in the S2/S3 pool. The score is macro F0.5 per S1, so precision
counts twice as much as recall, and an S1 with no true match only scores if nothing is predicted. Two facts from the data
shaped the design:
- **One-to-one ownership:** in the training labels no pool record belongs to more than one S1. Every pool record is given
  to at most one S1 (the highest-scoring one).
- **France is an unseen country** (test only) with template-like names ("Nantes Club SARL"). Many S1s share a name in the
  same city and have "sibling" businesses at the same address that differ by one descriptive word or legal form.

## 2. Candidate generation (blocking)
1. **Dense retrieval.** A multilingual MiniLM bi-encoder, contrastively fine-tuned on matched pairs with in-batch
   same-country negatives, embeds every record. GPU top-k search in blocks gives each S1 its nearest pool records in its country.
2. **Lexical retrieval** (France). A hashed TF-IDF over name tokens, character 4-grams, address tokens and house numbers.
   It is order-insensitive and language-agnostic, so it recovers French variants the India/US-trained retriever misses
   (legal form moved to the front, reordered words). Pool coverage in France rose from 79% (dense top-8) to 88% (+ lexical top-8).
3. **Exact-key blocking** (France). Hash keys on normalised name + house number + street (plus domain-handle, acronym and
   pseudo-name keys). Constant cost per record, so it scales to billions of records.

`candidate_pairs.tsv` holds the dense top-8 per S1, the lexical top-8 (France), the key-blocking pairs, and for India/US any
accepted pair from retrieval ranks 9-20. That is 8.78 candidates per S1 (15.2M pairs), and every matched pair is included.

## 3. Pair scoring (India / US)
- **63 hand-crafted features:** fuzzy name/address similarities on raw and normalised text, transliteration skeletons for
  Indic scripts, parsed street / house-number agreement, address-component checks, and contention (how many S1s retrieve
  the same pool record and how they rank).
- **Two cross-encoders as features** (MiniLM, 8M pairs; XLM-R base, 8M pairs). They are trained on S1 records disjoint from
  the LightGBM training sample, so their scores are out-of-sample. Features include the logit, the rank in the query and the
  gap to the best candidate.
- **Stage 1:** LightGBM on a 150k-S1-per-country sample.
- **Stage 2:** a second LightGBM using out-of-fold stage-1 probabilities plus per-query context (rank, gap to top-1,
  similarity to the top-1 and to other confident candidates, S1-name frequency, computed on a test-sized sample to avoid
  train/test shift).
- **Decision rule:** a gate threshold for the top candidate and a separate threshold for extra matches, tuned on held-out S1
  for exact macro F0.5; then one owner per pool record.
- **Validation:** held-out queries (group split by S1) matched the leaderboard throughout. India/US scores 0.9894 / 0.9869.

## 4. France (no labels)
- **Diagnosis via a leak-free rehearsal:** train on US only and evaluate on India as a stand-in unseen country. Hand
  features alone lose ~0.07. Any cross-encoder trained without the target country lowers F0.5 further (MiniLM 0.853,
  XLM-R 0.869 vs 0.903 without); larger models, letter-substitution augmentation and synthetic data all failed.
  France therefore uses **no text-model features**.
- **Model:** LightGBM on the country-independent features, **self-trained** on confident France pseudo-labels
  (P ≥ 0.95 / ≤ 0.05, two rounds). The rehearsal gain was +0.015, with thresholds chosen on India/US only.
  A stage-2 context model adds +0.004.
- **Rule-based post-processing,** each rule first checked for precision on India/US training labels:

| Rule (France only) | Train precision US / India | Effect |
|---|---|---|
| Drop extra matches whose name swaps a descriptive word (sibling pattern) | — (≈50% true in France) | LB +0.0023 |
| Empty-address record whose normalised name matches exactly one S1 | 96.6% / 97.5% | LB +0.0005 |
| Exact name, same house number and street, unique owner | 99.95% / 99.0% | LB +0.0014 |
| Typo-tolerant name / one word dropped, same address | 99.7–99.9% / 96.9–98.5% | LB +0.0003 |
| No house number / domain / acronym / invented trading name at the unique S1's address | 98.3–100% | LB +0.0009 |

- **Rejected after testing:** the same rules on India/US (the strong model already rejects those pairs correctly;
  leaderboard −0.0008), house-number-shift additions, word additions (India 72%), "any name at the address" (68–79%).

## 5. Leaderboard progression
0.966 → 0.969 (street / house-number features) → 0.975 (K = 20, stage 2, France self-training) → 0.976 (XLM-R) →
0.978 (France lexical candidates) → 0.980 (descriptor filter, empty-address rule) → 0.9816 (exact-key blocking) →
0.9819 (typo / word-drop) → 0.98275 (no-number, domain, acronym, pseudo-name keys) → **0.983** final leaderboard.

## 6. Reproduction
See the repository `README.md` for the architecture diagram and the exact commands and order. Every step writes checkpointed
artifacts, and `validate.py` checks the format of both output files.
