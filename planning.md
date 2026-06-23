# TakeMeter — Project Planning

---

## 1. Community

**Community:** r/AmItheAsshole (AITA)

AITA is a subreddit where users describe a specific interpersonal conflict and ask the community to judge who behaved wrongly. Every post ends in a community verdict — YTA (You're the Asshole), NTA (Not the Asshole), ESH (Everyone Sucks Here), NAH (No Assholes Here), or INFO — applied by moderator flair after community voting.

**Why this community fits a classification task:**

- **Discourse is varied by design.** Posts cover wildly different scenarios (family conflict, workplace dispute, romantic relationships, finances) but all share the same underlying question. The surface text varies enormously; the label space is fixed. That gap — diverse text, constrained labels — is exactly what a classifier needs to learn.
- **Labels are community-assigned, not externally imposed.** The verdicts exist because AITA participants invented and agreed on them. They reflect a real social norm within the community: these distinctions matter to actual participants.
- **Labels are pre-applied.** Each post has a community verdict flair that serves as a ground-truth label. This lets us collect labeled data directly from Reddit without manual annotation from scratch.
- **The task is genuinely hard.** Predicting moral verdicts requires understanding context, proportionality, and who initiated the conflict — not just keyword matching. A zero-shot baseline won't trivially solve it, and a fine-tuned model that learns the community's specific norms will measurably outperform one that doesn't.

---

## 2. Label Taxonomy

**Label 1: NTA — Not the Asshole**

*Definition:* The poster's behavior was justified given the situation; the conflict was initiated or primarily caused by the other party, and the poster's response was proportionate.

**Example 1:** "My roommate ate my clearly labeled food for the third time, so I bought a lock for my mini-fridge. He's furious. AITA?" → NTA. Setting a physical boundary in response to three repeated violations is proportionate; the roommate's anger does not change that the poster is reacting to ongoing harm.

**Example 2:** "I declined to lend my car to a friend who has two DUIs on her record. She's calling me selfish. AITA?" → NTA. Refusing a request that carries real personal and legal risk is justified regardless of the friend's reaction.

---

**Label 2: YTA — You're the Asshole**

*Definition:* The poster caused unjustified harm or acted unreasonably; the poster's action either initiated the conflict or constituted a disproportionate escalation relative to what was done to them.

**Example 1:** "I told my sister her baby is ugly at her baby shower because I believe in honesty. She asked me to leave. AITA?" → YTA. Causing public distress at a celebratory event for no constructive purpose is unjustified regardless of the poster's stated value of honesty.

**Example 2:** "I read my teenage daughter's private diary because I had a vague feeling something was wrong. I found nothing. She's devastated. AITA?" → YTA. Invading a minor's privacy without specific evidence of risk, and then finding nothing to justify it, is an unjustified harm.

---

**Label 3: ESH — Everyone Sucks Here**

*Definition:* Both the poster AND the other party behaved poorly in the same incident; there is no clear moral winner because multiple people made harmful or unreasonable choices.

**Example 1:** "My partner forgot our anniversary, so I canceled our joint vacation without telling them until the day before. They exploded. AITA?" → ESH. Forgetting an anniversary is hurtful; canceling a joint vacation without communication is a separate, disproportionate harm. Both parties failed.

**Example 2:** "My neighbor plays loud music at night. I started blasting music at 6am on weekends to make a point. Now we're both complaining to the landlord. AITA?" → ESH. The neighbor's late-night noise is inconsiderate; the poster's retaliation at 6am creates a new harm rather than resolving the original one. Both are wrong.

---

## 3. Hard Edge Cases

**Hardest anticipated boundary: YTA vs. ESH**

The scenario: the poster did something wrong, but the other party also behaved badly in the same incident. This is the most common source of genuine ambiguity in AITA.

**Example post at the boundary:** "My sister borrowed money and never paid it back. Last month at Christmas dinner in front of our whole family, I demanded she pay me in front of everyone. She had a breakdown. AITA?"
- The sister broke a financial agreement → she did something wrong (supports ESH)
- The poster chose public shaming at a family event → disproportionate and harmful (supports YTA)

**Decision rule:** If the poster's action was the **primary initiating harm in this specific incident** — meaning others were reacting to what the poster did — label **YTA**. If both parties independently caused harm at roughly equal severity with no single initiator, label **ESH**. In the example above: the sister's debt came first, but the public confrontation was the primary harm in this incident, and it was grossly disproportionate to the remedy. → **ESH** (the financial wrong does not justify public humiliation; both parties are at fault in this incident).

**Secondary boundary: NTA vs. ESH**

When a poster responds to ongoing mistreatment with a single retaliatory act. Decision rule: if the poster's response was proportionate in severity and scope (private to private, minor to minor), and it was a first-time response rather than ongoing retaliation → NTA. If the poster's response exceeded the original harm in scope or formality (e.g., public retaliation for a private slight), → ESH.

**When encountering ambiguous cases during annotation:** Apply the decision rule. If it still feels 50/50 after applying the rule, take the community flair as ground truth — the community's vote is the label, not our personal judgment.

---

## Hard Cases Found During Annotation (from actual data)

### Hard Case 1 — ESH vs. NTA (community verdict: ESH)

**Post summary:** A poster who was bullied for years by a classmate made fun of the bully's large nose in retaliation. The bully turned out to be Jewish and went around calling the poster antisemitic. The poster later apologized — but the apology was backhanded: "I didn't know he was a Jew and I wouldn't have joked about his nose if he hadn't spent years calling me slurs."

**Ambiguous between:** ESH and NTA

**What I decided:** ESH — aligns with the community flair. Even though the bully was the original aggressor and had spent years mistreating the poster, making a physical appearance joke (that could carry ethnic connotations regardless of intent) is still causing harm. The backhanded apology also shows the poster would have considered the joke acceptable had the bully not been Jewish — which undermines a clean NTA. Both parties behaved poorly in this incident.

**Why it's hard:** The bully had genuinely mistreated the poster for years, making NTA feel sympathetic. The decision rule says: if the poster's response was proportionate and it was a first-time response → NTA. Here, the poster's response was not proportionate (a slur rebuttal with an appearance joke that carries ethnic weight) and the apology revealed conditional judgment. → ESH.

---

### Hard Case 2 — YTA vs. NTA (community verdict: YTA)

**Post summary:** A poster (WIBTA framing) had held off introducing her boyfriend to her belittling father. When they finally met, the boyfriend defended the poster against her dad's put-downs. The poster now wants to ask the boyfriend to apologize to her dad for defending her — to keep family peace.

**Ambiguous between:** YTA and NTA

**What I decided:** YTA — aligns with the community flair. Asking someone who defended you to apologize to the person who was attacking you validates the attacker's behavior and punishes the defender. The poster's discomfort with conflict doesn't justify this. The "keeping peace" framing is understandable but the action itself is harmful to the boyfriend.

**Why it's hard:** The poster knows her family dynamic intimately and may have good reasons to manage confrontation carefully. A case could be made for NTA ("she's navigating a difficult family, it's not her fault she has to manage her dad"). But the action directly harms the boyfriend for doing something good — that tips it to YTA under the decision rule: the poster's action would constitute a disproportionate and unjustified harm to the defender.

---

### Hard Case 3 — ESH vs. NTA (community verdict: ESH)

**Post summary:** A poster in a masters biology class group project has been carrying the team — other members have been contributing poorly or not at all. The poster wants to stop doing any further work for the group going forward, as a protest.

**Ambiguous between:** ESH and NTA

**What I decided:** ESH — aligns with the community flair. If the poster unilaterally stops contributing without first escalating to the professor or having a direct conversation with the group, refusing to work also harms the project and potentially innocent group members. The teammates' poor contribution is wrong, but withdrawing labor entirely rather than escalating is also a harmful choice.

**Why it's hard:** If the poster genuinely tried to resolve the imbalance and was ignored, a unilateral work stoppage starts to look like NTA (forced action against an unresponsive situation). The post doesn't mention prior escalation — and the absence of that context is what makes it ESH rather than NTA. Decision rule: if both parties independently caused harm with no clear resolution attempt → ESH.

---

## 4. Data Collection Plan

**Source:** Public HuggingFace AITA dataset (pre-labeled with community verdict flairs — no API credentials required, loaded via `datasets.load_dataset()`). Exact dataset name verified in Colab at the data-loading step.

**Collection process:**
1. Load the full dataset from HuggingFace
2. Filter to rows with verdict in `{YTA, NTA, ESH}` — drop INFO and NAH (combined <5%, would create severe class imbalance)
3. Stratified sample ~250 rows to account for cleaning losses
4. Clean: strip Reddit markdown, URLs, `>quote` blocks; drop rows <20 words after cleaning
5. Target distribution: ~40% NTA (~92), ~35% YTA (~80), ~25% ESH (~58); total ~230 clean examples
6. Split (stratified, seed=42): **140 train / 20 validation / 70 test**

**If a label is underrepresented after filtering:**
- If ESH falls below 45 clean examples (20% of 230): over-sample from the raw dataset by relaxing filters (e.g., include posts with slightly fewer words) until the threshold is met.
- If the HuggingFace dataset doesn't yield enough ESH: supplement with direct Reddit PRAW collection filtered to ESH-flaired posts.
- If the distribution can't be balanced to ≥20% per class: reduce to 2 labels (merge YTA and ESH into a single "poster bears responsibility" class) and document the decision in README. This is a last resort.

**Count per label (target):** 92 NTA / 80 YTA / 58 ESH = 230 total.

---

## 5. Evaluation Metrics

**Metrics used and why:**

| Metric | Why it's necessary |
|---|---|
| **Overall accuracy** | Provides a single-number summary for comparison between models; required by project spec |
| **Per-class F1 (precision + recall)** | Accuracy alone is misleading here: a model that always predicts NTA gets 40% accuracy but is completely useless. Per-class F1 reveals whether the model actually learns each class or ignores minority classes |
| **Macro F1** | Averages F1 across all classes with equal weight regardless of class size — this penalizes the model for ignoring ESH (the smallest class) and is the primary metric for comparing fine-tuned vs. baseline |
| **Confusion matrix (3×3)** | Shows which pairs of labels are confused. The expected failure mode is YTA↔ESH confusion (both involve the poster doing something wrong); the matrix makes this visible |

**Why accuracy alone isn't enough:**
With a 40/35/25 distribution, a degenerate model that always predicts NTA achieves 40% accuracy — higher than a model that actually tries to distinguish all three classes but makes some mistakes. Macro F1 for the NTA-only model is ~13%, correctly exposing the failure. Accuracy would make it look competitive.

**Primary comparison metric:** Macro F1 (fine-tuned vs. zero-shot baseline). Accuracy reported alongside it.

---

## 6. Definition of Success

**Minimum threshold (model is genuinely useful):**
- Fine-tuned model: ≥65% accuracy AND ≥0.55 macro F1 on the 70-example test set
- Fine-tuned model outperforms zero-shot Groq by ≥5 percentage points on macro F1
- ESH recall ≥ 0.35 (model predicts ESH on at least 35% of true ESH examples — it didn't collapse to only predicting NTA/YTA)

**Rationale for these thresholds:**
- 65% accuracy is meaningfully above the "always predict NTA" baseline (40%) and above random (33%)
- 0.55 macro F1 means the model is performing reasonably on all three classes, not just the majority
- The ≥5pp macro F1 gap over Groq justifies the fine-tuning effort
- ESH recall ≥0.35 ensures the model actually learned the minority class

**"Good enough for deployment" in a real community tool:**
≥75% accuracy AND ≥0.65 macro F1 AND ESH F1 ≥ 0.50. At this level, the model correctly identifies the verdict most of the time and doesn't systematically ignore any class.

**What would indicate failure:**
- Accuracy ≤50% (barely above majority-class baseline)
- ESH recall = 0 (model never predicts ESH — learned nothing about the minority class)
- Fine-tuned model performs worse than or equal to zero-shot Groq (fine-tuning added no signal)
- Accuracy >95% (likely data leakage from test set into training)

---

## 7. AI Tool Plan

### Label Stress-Testing (before annotation)

**Plan:** Give Claude the three label definitions and the YTA/ESH edge case description, and ask it to generate 10 posts that sit at the boundary between two labels. If Claude produces posts I can't cleanly classify using the decision rules, I tighten the rules before annotating 200 examples.

**Prompt template:**
```
Here are my AITA label definitions:
- NTA: [definition]
- YTA: [definition]
- ESH: [definition]

Decision rule for YTA vs ESH: [rule]

Generate 10 Reddit posts that sit at the hardest possible boundary between YTA and ESH — posts where a reasonable person could argue for either label. Make them realistic, varied in scenario, and as ambiguous as possible.
```

**Success criterion:** I can apply the decision rule to all 10 posts and arrive at a confident label in under 30 seconds each. If more than 2 posts still feel 50/50 after applying the rule, the rule needs to be revised.

### Annotation Assistance

**Decision: No LLM pre-labeling.** The data source (HuggingFace AITA dataset) already has community verdict flairs as ground-truth labels. The "labeling process" is: load the flair, map it to {NTA, YTA, ESH}, review ~30 examples manually to verify the flair matches the post content, and document any mismatches. LLM annotation would introduce noise on top of community ground truth and is not needed.

**What I'll do instead:** Manually review 30 sampled examples (10 per class) to verify label quality and identify any posts where the flair appears to be wrong. Document the review in README.

### Failure Analysis (after evaluation)

**Plan:** After running evaluation, give Claude the complete list of wrong predictions (post text + true label + predicted label) and ask it to identify systematic patterns before writing up the analysis.

**Prompt template:**
```
Here are the test examples my classifier got wrong. Each entry shows the post text, the true label, and what the model predicted.

[list of errors]

Identify any systematic patterns in these errors. Look for: common scenarios, post lengths, writing styles, or specific label confusions that appear more than once. Group the errors into 2-3 pattern categories if possible.
```

**Verification:** After Claude identifies patterns, I manually count the examples in each pattern category to verify the pattern explains ≥30% of errors before including it in the evaluation report. Claude's pattern is a hypothesis; my count is the evidence.

---

## 8. Model Plan

- **Base model:** `distilbert-base-uncased` (HuggingFace)
- **Fine-tuning:** HuggingFace Trainer API in Google Colab (free T4 GPU)
- **Key hyperparameter decisions:**
  - Learning rate 2e-5: standard for DistilBERT; lower rates underfit on 140 examples
  - 5 epochs with early stopping (patience=2, metric=val_accuracy): small dataset benefits from more passes with validation-based stopping
  - Batch size 16: fits T4 memory; smaller batches add noise without benefit at this scale
  - Max sequence length 512: AITA posts are often long; truncating earlier loses context

---

## 9. Baseline Plan

- **Model:** Groq `llama-3.3-70b-versatile`, zero-shot (no task-specific training)
- **Prompt:** System instruction + post text; model asked to respond with exactly one word (YTA, NTA, or ESH)
- **Purpose:** Measure how much fine-tuning actually improves over a strong zero-shot LLM

---

## 10. Stretch Features

- [ ] **Deployed interface:** Gradio UI in Colab — input a post, get label + confidence scores per class
- [ ] **Error pattern analysis:** Systematically categorize all test-set errors; identify if ≥1 pattern explains ≥30% of failures (see AI Tool Plan above)

---

## 11. Validation Standards

Every pipeline step verified before proceeding:
- No data leakage between train/val/test splits (assert disjoint index sets in code)
- Label distribution printed and verified at each pipeline stage
- Model tested on ≥3 hand-crafted posts before running formal evaluation
- Groq responses logged raw before parsing — no silent failures
- All metrics computed via sklearn to avoid implementation bugs
- Confusion matrix axis labels explicitly set to `[NTA, YTA, ESH]` to prevent axis scrambling
- Test set evaluated exactly once, at the very end
