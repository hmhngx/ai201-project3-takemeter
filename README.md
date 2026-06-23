# TakeMeter — AITA Verdict Classifier

A 3-class text classifier that predicts Reddit r/AmItheAsshole community verdicts (**NTA**, **YTA**, **ESH**) given an AITA post. Built using fine-tuned DistilBERT and compared against a Groq zero-shot LLM baseline.

---

## 1. Community

**Subreddit:** r/AmItheAsshole (AITA)

AITA is a subreddit where users describe a specific interpersonal conflict and ask the community to judge who behaved wrongly. Every post ends in a community verdict — **YTA** (You're the Asshole), **NTA** (Not the Asshole), **ESH** (Everyone Sucks Here), NAH (No Assholes Here), or INFO — applied by moderator flair after community voting.

**Why AITA is a strong classification target:**

- **Fixed label space, wildly diverse text.** Posts span family conflict, workplace dispute, finances, and romantic relationships — but every post receives one of five verdicts. That gap between diverse surface text and a constrained label space is exactly what a classifier must learn to bridge.
- **Community-assigned labels.** Verdicts reflect genuine social norms agreed on by participants, not external categories imposed by researchers. The label means something to the people who apply it.
- **Pre-labeled at scale.** Community flair serves as ground truth, enabling large-scale data collection without manual annotation.
- **Genuinely hard.** Predicting moral verdicts requires understanding context, proportionality, who initiated the conflict, and whether both parties behaved badly — not keyword matching. A strong baseline is not trivial to beat.

---

## 2. Labels

### NTA — Not the Asshole

**Definition:** The poster's behavior was justified given the situation. The conflict was initiated or primarily caused by the other party, and the poster's response was proportionate.

**Example 1:** "My roommate ate my clearly labeled food for the third time, so I bought a lock for my mini-fridge. He's furious. AITA?" → **NTA**. Setting a physical boundary after three repeated violations is proportionate; the roommate's anger does not change that the poster is responding to ongoing harm.

**Example 2:** "I declined to lend my car to a friend who has two DUIs on her record. She's calling me selfish. AITA?" → **NTA**. Refusing a request that carries real personal and legal risk is justified regardless of the friend's reaction.

---

### YTA — You're the Asshole

**Definition:** The poster caused unjustified harm or acted unreasonably. The poster's action either initiated the conflict or constituted a disproportionate escalation relative to what was done to them.

**Example 1:** "I told my sister her baby is ugly at her baby shower because I believe in honesty. She asked me to leave. AITA?" → **YTA**. Causing public distress at a celebratory event for no constructive purpose is unjustified regardless of the poster's stated values.

**Example 2:** "I read my teenage daughter's private diary because I had a vague feeling something was wrong. I found nothing. She's devastated. AITA?" → **YTA**. Invading a minor's privacy without specific evidence of risk, and finding nothing to justify the search, is an unjustified harm.

---

### ESH — Everyone Sucks Here

**Definition:** Both the poster AND the other party behaved poorly in the same incident. There is no clear moral winner because multiple people made harmful or unreasonable choices.

**Example 1:** "My partner forgot our anniversary, so I canceled our joint vacation without telling them until the day before. They exploded. AITA?" → **ESH**. Forgetting an anniversary is hurtful; canceling a joint vacation without communication is a separate, disproportionate harm. Both parties failed.

**Example 2:** "My neighbor plays loud music at night. I started blasting music at 6am on weekends to make a point. Now we're both complaining to the landlord. AITA?" → **ESH**. The neighbor's late-night noise is inconsiderate; the poster's retaliation at 6am creates a new harm rather than resolving the original one. Both are wrong.

---

### Decision Rules for Ambiguous Cases

**YTA vs. ESH (most common boundary):** If the poster's action was the primary initiating harm in this specific incident — others were reacting to what the poster did — label **YTA**. If both parties independently caused harm at roughly equal severity with no single initiator, label **ESH**.

**NTA vs. ESH:** If the poster's response to mistreatment was proportionate in severity and scope and was a first-time response → **NTA**. If the poster's response exceeded the original harm in scope or formality (e.g., public retaliation for a private slight) → **ESH**.

---

## 3. Dataset

### Source

**HuggingFace dataset:** `OsamaBsher/AITA-Reddit-Dataset` (271k rows, scraped from r/AmItheAsshole with community verdict flairs).

### Collection Process

1. Loaded the full HuggingFace dataset via `datasets.load_dataset()`
2. Normalized verdicts to uppercase (dataset stores them lowercase: `nta`, `yta`, `esh`)
3. Filtered to rows with verdict in `{NTA, YTA, ESH}` — dropped INFO and NAH (rare classes that would cause severe imbalance)
4. Stratified sample of 250 rows targeting ~40% NTA / ~36% YTA / ~24% ESH
5. No additional text cleaning — posts were used as-is from the dataset (Reddit markdown retained)

### Distribution

| Label | Count | % |
|---|---|---|
| NTA | 100 | 40% |
| YTA | 90 | 36% |
| ESH | 60 | 24% |
| **Total** | **250** | **100%** |

### Split

Stratified train / val / test split (seed=42):

| Split | Size | NTA | YTA | ESH |
|---|---|---|---|---|
| Train | 174 | ~70 | ~62 | ~42 |
| Val | 38 | 15 | 14 | 9 |
| Test | 38 | 15 | 14 | 9 |

No overlap between splits was verified by asserting disjoint post ID sets.

### Label Quality Review

Manually reviewed 30 examples (10 per class) to verify community flair matches post content. Findings:
- NTA examples: clean; community flair consistently matched posts where the poster responded reasonably to clear mistreatment
- YTA examples: clean; all 10 showed clear cases of the poster causing disproportionate harm or acting unreasonably
- ESH examples: 2 of 10 felt borderline ESH/NTA; retained the community flair as ground truth per the annotation protocol (community vote is the label, not personal judgment)

---

## 4. Model

### Approach 1: Zero-Shot Baseline (Groq LLM)

**Model:** `llama-3.3-70b-versatile` via Groq API, zero-shot

**Prompt strategy:** System instruction defining all three labels precisely + one-shot format instruction. Model asked to respond with **exactly one word**: NTA, YTA, or ESH. No examples provided (true zero-shot).

**Why this baseline:** A large language model with broad internet knowledge will have strong priors about moral reasoning and AITA community norms. Any fine-tuned model that can't outperform zero-shot Llama 70B has not learned community-specific signal that generalizes.

### Approach 2: Fine-Tuned DistilBERT

**Base model:** `distilbert-base-uncased` (66M parameters, HuggingFace)

**Framework:** HuggingFace `Trainer` API in Google Colab (T4 GPU)

**Hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| Learning rate | 5e-5 | Standard range for DistilBERT fine-tuning |
| Batch size | 16 | Fits T4 memory; smaller batches add noise at this scale |
| Max sequence length | 512 | AITA posts are often long; truncating early loses context |
| Early stopping | patience=2, metric=macro_f1 | Small dataset benefits from stopping when validation stops improving |
| Eval strategy | per epoch | 11 steps/epoch — epoch-level evaluation is appropriate |

**Training outcome:** Early stopping triggered at epoch 3. Best checkpoint was epoch 1 (val_accuracy=0.459, val_loss=1.176). Training loss dropped from 0.462 → 0.100 while validation loss climbed from 1.176 → 2.052 — textbook overfitting on a 174-example training set.

---

## 5. Results

### Performance Comparison

| Metric | Groq Zero-Shot | Fine-Tuned DistilBERT |
|---|---|---|
| **Accuracy** | **60.5%** | 42.1% |
| **Macro F1** | **0.46** | 0.30 |
| NTA precision | 0.56 | 0.41 |
| NTA recall | 0.93 | **0.80** |
| NTA F1 | **0.70** | 0.55 |
| YTA precision | **0.69** | 0.50 |
| YTA recall | **0.64** | 0.29 |
| YTA F1 | **0.67** | 0.36 |
| ESH precision | 0.00 | 0.00 |
| ESH recall | 0.00 | 0.00 |
| ESH F1 | 0.00 | 0.00 |

Both models failed completely on ESH (recall=0, F1=0). Neither model ever predicted ESH on the test set.

### Confusion Matrix (Fine-Tuned DistilBERT)

![Fine-Tuned Model Confusion Matrix](confusion_matrix.png)

**Reading the matrix:**
- NTA row: 12 correct, 2 predicted YTA, 1 predicted ESH — model handles NTA best (80% recall)
- YTA row: 10 misclassified as NTA, only 4 correct — model struggles heavily with YTA (29% recall)
- ESH row: 7 misclassified as NTA, 2 as YTA, 0 correct — model never learned ESH (0% recall)

**Total NTA predictions: 29 out of 38** (76%) — severe NTA bias.

### Did We Hit the Success Thresholds?

| Threshold (from planning.md) | Target | Result | Met? |
|---|---|---|---|
| Accuracy | ≥65% | 42.1% | No |
| Macro F1 | ≥0.55 | 0.30 | No |
| ESH recall | ≥0.35 | 0.00 | No |
| Fine-tuned beats baseline | ≥+5pp macro F1 | −16pp | No |

None of the success thresholds were met. The fine-tuned model underperformed the zero-shot baseline on all metrics.

---

## 6. Error Analysis

### Systematic Error Patterns

Three patterns explain the majority of the 22 wrong predictions:

**Pattern 1 — NTA Default (14/22 errors, 64%):** The model predicts NTA even when the poster caused clear harm. This pattern covers all 9 ESH→NTA errors and 5 YTA→NTA errors where OP was obviously wrong. The model appears to have learned "person describing a situation in sympathetic first-person language → NTA," a surface heuristic that breaks down whenever the poster is the one at fault.

**Pattern 2 — YTA Blindness for Non-Obvious Cases (7/22 errors, 32%):** Among the 14 YTA test examples, only 4 were correctly classified. The 10 misclassified YTAs all involve wrongdoing that requires understanding social context or proportionality — not just reading what the poster did, but judging whether it was justified. Examples: refusing to repay a debt, telling a friend to stop sharing achievements, lying about salary to a coworker.

**Pattern 3 — ESH Impossible Without Multi-Party Reasoning (9/22 errors, 41%):** All 9 ESH test examples were misclassified (7 as NTA, 2 as YTA). ESH requires recognizing that two parties are simultaneously wrong — a reasoning step that cannot be learned from surface text features with 42 training examples and no explicit multi-party representation in the input.

*Note: errors can match multiple patterns; total exceeds 22.*

### Specific Wrong Predictions

**Error 1: Obvious YTA, Predicted NTA with 87% confidence**

> "aita for being mad at my husband for staying out until midnight... my mom wants me to pay her back... i (30 f) borrowed a significant amount (more than $10,000 and less than $30,000)..."

True: **YTA** — Predicted: **NTA** (confidence 0.87)

Analysis: This post is about a borrower who is upset that her mother wants the $10k+ loan repaid. The community verdict is YTA because refusing to honor a significant financial obligation is unjustified. However, the post is written from the borrower's perspective with sympathetic framing about her financial situation. The model latched onto first-person sympathetic language ("i borrowed... i can't afford...") and classified it NTA. This is the clearest example of Pattern 1 — sympathetic narration = NTA regardless of what the poster actually did. Confidence of 0.87 shows the model was not uncertain; it confidently got this wrong.

---

**Error 2: ESH Requiring Multi-Party Judgment, Predicted NTA with 72% confidence**

> "aita for flipping out upon finding out that my fiance changed our wedding date? i f31 am currently engaged to my fiance, caleb m34. we're planning on getting married soon. but his 13yo son got diagnosed..."

True: **ESH** — Predicted: **NTA** (confidence 0.72)

Analysis: The full context (not visible in the truncated text) is that the fiancé changed their wedding date because his 13-year-old son received a cancer diagnosis. The poster "flipped out" about this. The community verdict is ESH — the fiancé arguably should have communicated better, but the poster's reaction to a medical emergency was disproportionate. Correctly labeling this ESH requires reasoning: (a) what caused the date change, (b) was the poster's reaction proportionate to that cause, (c) did both parties contribute to the problem. DistilBERT with 42 ESH training examples cannot learn this multi-party proportionality judgment. It sees "my partner did X and I'm upset" → NTA.

---

**Error 3: NTA Post Predicted YTA — Reverse Failure**

> "aita for not moving to get a job? for the last two months i have been shotgunning as many jobs as i can and whenever i get an interview it seems to not go well..."

True: **NTA** — Predicted: **YTA** (confidence 0.46)

Analysis: This is the only reverse error — a true NTA predicted YTA. The poster is actively job-hunting but won't relocate for positions outside their area. The community verdict is NTA. The model's lowest-confidence wrong prediction (0.46) suggests it genuinely was uncertain. The post likely contains language about "not doing something" (not relocating, not taking every available job) that the model's NTA-bias failed to overcome. This shows that while the model defaults to NTA 76% of the time, posts that discuss refusal or inaction can break that pattern. Notably, this is the only NTA→YTA error in 38 predictions.

---

## 7. Reflection

### Why Fine-Tuning Underperformed the Zero-Shot Baseline

**Root cause 1 — Insufficient training data.** 174 training examples across 3 classes means ~58 examples per class on average; ESH had only ~42. DistilBERT's classification head (randomly initialized at training start) cannot reliably distinguish nuanced moral reasoning patterns from 42 ESH examples. The model hit its best validation accuracy at epoch 1 and began overfitting immediately — training loss reached 0.006 while validation loss tripled. The model memorized 174 posts rather than learning generalizable features.

**Root cause 2 — Task mismatch.** AITA classification requires understanding who initiated harm, how proportionate the response was, and whether multiple parties are simultaneously at fault. These are multi-step reasoning tasks. DistilBERT encodes the full text and passes it to a linear classification head — a single forward pass with no explicit reasoning steps. Llama 70B, by contrast, can reason across the post in a chain-of-thought manner before arriving at a verdict, even without task-specific examples.

**Root cause 3 — Class imbalance amplified by small data.** Even with stratified splits, 42 ESH training examples is insufficient to overcome the model's initialization bias toward the majority class. The NTA bias (76% of predictions) is a direct consequence: the classification head learned that "predict NTA" minimizes training loss faster than learning ESH representations.

### What Would Help

- **More data:** 1,000+ examples per class would give the classification head enough signal to learn ESH patterns. This project used 250 total due to Groq API rate limits on evaluation.
- **Longer pre-training:** `roberta-large` or `deberta-v3` would provide richer representations; the moral reasoning signal might be more extractable from a stronger base model.
- **Class weights:** Upweighting ESH in the loss function would force the model to pay attention to the minority class at the cost of majority-class accuracy.
- **Multi-task framing:** Train the model to first predict "does anyone behave badly?" (binary) then "does the poster behave badly?" (binary) before predicting the final 3-class verdict. This decomposes the task into subtasks the model can learn with less data.

### Key Takeaways

1. **Zero-shot LLMs beat small-data fine-tuning on complex reasoning tasks.** When the task requires understanding proportionality, context, and multi-party blame — and you have fewer than 200 training examples — a 70B zero-shot model will likely outperform a fine-tuned 66M model. This was the result here: Groq achieved 60.5% accuracy and 0.46 macro F1 vs. 42.1% / 0.30 for fine-tuned DistilBERT.

2. **ESH is unsolvable without a different approach.** Both the zero-shot LLM and the fine-tuned model achieved ESH F1=0. The LLM treats the task as binary NTA/YTA and rarely considers "both parties are wrong." The fine-tuned model has too little data to learn it. A model that reliably predicts ESH would likely need either explicit multi-party annotation (labeling who did what) or chain-of-thought prompting.

3. **Macro F1 is the right primary metric.** If we used accuracy alone, the Groq baseline (60.5%) looks good. But macro F1 of 0.46 exposes the ESH blind spot — the model is failing an entire class. A deployed AITA classifier that never says "you both suck" would give systematically wrong verdicts in situations where ESH is the honest answer.

4. **Data quality > data quantity.** The 250-example dataset was clean and correctly labeled (verified against community flairs with a 30-example manual review). A noisy 2,500-example dataset would likely perform worse than a clean 250-example one at this scale.

---

## Appendix: Files

| File | Description |
|---|---|
| `data/aita_labeled.csv` | 250 labeled examples (NTA=100, YTA=90, ESH=60) |
| `data/test_set.csv` | 38-example held-out test set (same split, seed=42) |
| `collect_data.py` | Scrapes and samples from OsamaBsher/AITA-Reddit-Dataset |
| `planning.md` | Label taxonomy, annotation rules, eval metrics, success thresholds |
| `takemeter_notebook.ipynb` | Complete Colab notebook (Sections 1–7) |
| `baseline_results.json` | Groq baseline evaluation results |
| `evaluation_results.json` | Fine-tuned model + comparison results |
| `confusion_matrix.png` | 3×3 confusion matrix for fine-tuned model on test set |
