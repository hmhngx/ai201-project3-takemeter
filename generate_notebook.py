"""
Generates takemeter_notebook.ipynb — the complete Colab notebook for TakeMeter.
Upload this file to Colab (File -> Upload notebook) to replace the empty starter notebook.
"""
import json

def cell(source, cell_type="code"):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

cells = [

# ─────────────────────────────────────────────────────────────────────────────
cell("# TakeMeter — AITA Verdict Classifier\n\n"
     "Classify Reddit AITA posts as **NTA**, **YTA**, or **ESH** using fine-tuned DistilBERT.\n\n"
     "**Run order for each milestone:**\n"
     "- Milestone 4 (baseline): Sections 1 → 2 → 5\n"
     "- Milestone 5 (fine-tuning): Sections 1 → 2 → 3 → 4 → 6 → 7",
     "markdown"),

# ─────────────────────────────────────────────────────────────────────────────
cell("# Section 1: Setup and Data Upload", "markdown"),

cell("""\
# Install core dependencies
!pip install datasets scikit-learn groq gradio -q
print("Dependencies installed.")
"""),

cell("""\
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── Label map (must match planning.md exactly) ────────────────────────────────
LABEL2ID = {"NTA": 0, "YTA": 1, "ESH": 2}
ID2LABEL  = {0: "NTA", 1: "YTA", 2: "ESH"}
LABELS    = ["NTA", "YTA", "ESH"]   # fixed order for all metrics and plots
SEED      = 42
print("Label map defined:", LABEL2ID)
"""),

cell("""\
# Upload your labeled CSV from your computer
from google.colab import files
uploaded = files.upload()  # select data/aita_labeled.csv when prompted

import io
filename = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[filename]))
print(f"Uploaded: {filename}")
print(f"Rows: {len(df)}, Columns: {list(df.columns)}")

# Validate
assert "text"  in df.columns, "Missing 'text' column"
assert "label" in df.columns, "Missing 'label' column"
assert df["text"].isna().sum() == 0,  "NaN found in text"
assert df["label"].isna().sum() == 0, "NaN found in label"
assert set(df["label"].unique()) == set(LABEL2ID.keys()), \\
    f"Unexpected labels: {df['label'].unique()}"

df["label_id"] = df["label"].map(LABEL2ID)

print("\\nLabel distribution:")
print(df["label"].value_counts().to_string())
pcts = df["label"].value_counts(normalize=True) * 100
assert (pcts >= 20).all(), "A label is below 20% — check your CSV"
assert (pcts < 70).all(),  "A label exceeds 70% — severe imbalance"
print("Distribution check PASSED.")
"""),

# ─────────────────────────────────────────────────────────────────────────────
cell("# Section 2: Split and Tokenize", "markdown"),

cell("""\
# Train / Val / Test split  (70% / 15% / 15%, stratified)
df_trainval, df_test = train_test_split(
    df, test_size=0.15, random_state=SEED, stratify=df["label_id"]
)
df_train, df_val = train_test_split(
    df_trainval, test_size=0.15/0.85, random_state=SEED, stratify=df_trainval["label_id"]
)

print(f"Split sizes — train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

# No-leakage assertion
train_ids = set(df_train["id"].tolist())
val_ids   = set(df_val["id"].tolist())
test_ids  = set(df_test["id"].tolist())
assert len(train_ids & val_ids)  == 0, "LEAK: train/val overlap"
assert len(train_ids & test_ids) == 0, "LEAK: train/test overlap"
assert len(val_ids   & test_ids) == 0, "LEAK: val/test overlap"
print("No overlap between splits: PASSED")

for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
    assert set(split_df["label"].unique()) == set(LABEL2ID.keys()), \\
        f"{name} split is missing some labels"
    print(f"{name} distribution: {dict(split_df['label'].value_counts())}")
"""),

cell("""\
from transformers import DistilBertTokenizerFast
from datasets import Dataset

MODEL_NAME = "distilbert-base-uncased"
tokenizer  = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize_batch(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

def df_to_hf_dataset(df_split):
    ds = Dataset.from_pandas(
        df_split[["text", "label_id"]]
        .rename(columns={"label_id": "labels"})
        .reset_index(drop=True)
    )
    ds = ds.map(tokenize_batch, batched=True)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds

train_ds = df_to_hf_dataset(df_train)
val_ds   = df_to_hf_dataset(df_val)
test_ds  = df_to_hf_dataset(df_test)

print(f"Tokenized — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

# Spot-check: decode first training token sequence back to text
decoded = tokenizer.decode(train_ds[0]["input_ids"], skip_special_tokens=True)
assert len(decoded) > 10, "Tokenization returned empty output"
print(f"Spot-check (first 80 chars): {decoded[:80]}...")
print("Section 2 complete.")
"""),

# ─────────────────────────────────────────────────────────────────────────────
cell("# Section 3: Fine-Tuning (Milestone 5)", "markdown"),

cell("""\
# ── Run this section only after Milestone 4 (baseline) is complete ─────────
from transformers import (
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
print(f"Model loaded: {MODEL_NAME}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "macro_f1":  f1_score(labels, preds, average="macro"),
    }

training_args = TrainingArguments(
    output_dir            = "./checkpoints",
    num_train_epochs      = 5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 16,
    learning_rate         = 2e-5,
    warmup_ratio          = 0.1,
    weight_decay          = 0.01,
    eval_strategy         = "epoch",
    save_strategy         = "epoch",
    load_best_model_at_end   = True,
    metric_for_best_model    = "macro_f1",
    greater_is_better        = True,
    logging_steps         = 10,
    seed                  = SEED,
    report_to             = "none",
)

trainer = Trainer(
    model          = model,
    args           = training_args,
    train_dataset  = train_ds,
    eval_dataset   = val_ds,
    compute_metrics= compute_metrics,
    callbacks      = [EarlyStoppingCallback(early_stopping_patience=2)],
)

print("Starting training...")
trainer.train()
print("Training complete.")

trainer.save_model("./best_model")
tokenizer.save_pretrained("./best_model")
print("Model saved to ./best_model")
"""),

# ─────────────────────────────────────────────────────────────────────────────
cell("# Section 4: Smoke-Test Fine-Tuned Model (Milestone 5)", "markdown"),

cell("""\
# Run inference on 3 hand-crafted examples to confirm the model loaded correctly
from transformers import pipeline as hf_pipeline

verdict_clf = hf_pipeline(
    "text-classification",
    model="./best_model",
    tokenizer="./best_model",
    return_all_scores=True,
)

SMOKE_TESTS = [
    {"text": "My roommate ate my clearly labeled food for the third time. I bought a lock for my fridge. He's furious. AITA?",
     "expected": "NTA"},
    {"text": "I told my sister her baby was ugly at her baby shower because I believe in honesty. She asked me to leave. AITA?",
     "expected": "YTA"},
    {"text": "My neighbor plays loud music at night. I started blasting music at 6am to retaliate. Now we are both complaining to the landlord. AITA?",
     "expected": "ESH"},
]

print("Smoke test — 3 hand-crafted examples:")
all_passed = True
for case in SMOKE_TESTS:
    result    = verdict_clf(case["text"])[0]
    scores    = {r["label"]: round(r["score"], 3) for r in result}
    predicted = max(result, key=lambda x: x["score"])["label"]
    status    = "PASS" if predicted == case["expected"] else "WARN"
    print(f"[{status}] Expected={case['expected']}, Got={predicted}, Scores={scores}")
    if predicted != case["expected"]:
        all_passed = False

if all_passed:
    print("\\nAll 3 smoke tests PASSED.")
else:
    print("\\nWARNING: Some smoke tests failed — investigate before final evaluation.")
"""),

# ─────────────────────────────────────────────────────────────────────────────
cell("# Section 5: Groq Zero-Shot Baseline", "markdown"),

cell("""\
import os, time, json
from groq import Groq

# ── Add your Groq API key to Colab Secrets (left sidebar → key icon)
# Key name: GROQ_API_KEY
try:
    from google.colab import userdata
    GROQ_API_KEY = userdata.get("GROQ_API_KEY")
    print("Loaded GROQ_API_KEY from Colab Secrets.")
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    print("Loaded GROQ_API_KEY from environment variable.")

assert GROQ_API_KEY, "GROQ_API_KEY is empty — add it to Colab Secrets"
print(f"API key present: {GROQ_API_KEY[:8]}...")

client = Groq(api_key=GROQ_API_KEY)
"""),

cell("""\
# ── Classification prompt ─────────────────────────────────────────────────────
# Label definitions match planning.md exactly.
GROQ_PROMPT = \"\"\"You are classifying posts from r/AmItheAsshole.

Label definitions:
- NTA (Not the Asshole): The poster's behavior was justified; the conflict was initiated or primarily caused by the other party.
- YTA (You're the Asshole): The poster caused unjustified harm or acted unreasonably; the poster initiated or disproportionately escalated the conflict.
- ESH (Everyone Sucks Here): Both the poster AND the other party behaved poorly; no clear moral winner.

Instructions: Read the post and respond with EXACTLY ONE WORD — either NTA, YTA, or ESH. No explanation, no punctuation, just the label.

Post: {text}

Verdict:\"\"\"

def classify_with_groq(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user",
                           "content": GROQ_PROMPT.format(text=text[:2000])}],
                temperature=0,
                max_tokens=10,
            )
            raw = response.choices[0].message.content.strip().upper()
            for label in ["NTA", "YTA", "ESH"]:
                if label in raw:
                    return label, raw
            print(f"  WARN unparseable: {repr(raw)}")
            return None, raw
        except Exception as e:
            print(f"  ERROR attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
    return None, "API_ERROR"

print("Prompt defined. Ready to run baseline.")
"""),

cell("""\
# ── Run baseline on the locked test set ──────────────────────────────────────
# Uses df_test created in Section 2 — do NOT re-split
test_texts  = df_test["text"].tolist()
true_labels = df_test["label"].tolist()

print(f"Running Groq baseline on {len(test_texts)} test examples...")
groq_preds = []
groq_raws  = []
parse_failures = []

for i, text in enumerate(test_texts):
    pred, raw = classify_with_groq(text)
    groq_preds.append(pred)
    groq_raws.append(raw)
    status = "OK" if pred is not None else "FAIL"
    print(f"  [{i+1:02d}/{len(test_texts)}] raw={repr(raw)[:30]:<32} parsed={pred}  [{status}]")
    time.sleep(0.5)   # stay within Groq free-tier rate limit

failures = [i for i, p in enumerate(groq_preds) if p is None]
print(f"\\nParsed: {len(groq_preds)-len(failures)}/{len(groq_preds)}")
if failures:
    print(f"Failed rows: {failures}")
    print("Treating failed rows as NTA (majority class) — they count as wrong predictions.")
    groq_preds = [p if p is not None else "NTA" for p in groq_preds]

df_test_results = df_test.copy().reset_index(drop=True)
df_test_results["groq_pred"] = groq_preds
df_test_results["groq_raw"]  = groq_raws
"""),

cell("""\
from sklearn.metrics import accuracy_score, f1_score, classification_report

groq_acc      = accuracy_score(true_labels, groq_preds)
groq_macro_f1 = f1_score(true_labels, groq_preds, average="macro", labels=LABELS)

print("=" * 60)
print("GROQ ZERO-SHOT BASELINE RESULTS")
print("=" * 60)
print(f"  Overall accuracy : {groq_acc:.4f}  ({groq_acc*100:.1f}%)")
print(f"  Macro F1         : {groq_macro_f1:.4f}")
print()
print(classification_report(true_labels, groq_preds, labels=LABELS))

# Save for comparison later
baseline_results = {
    "model": "llama-3.3-70b-versatile (zero-shot)",
    "accuracy":   groq_acc,
    "macro_f1":   groq_macro_f1,
    "per_class":  classification_report(true_labels, groq_preds, labels=LABELS, output_dict=True),
    "parse_failures": len(failures),
}
with open("baseline_results.json", "w") as f:
    json.dump(baseline_results, f, indent=2)
print("Saved baseline_results.json")

# Reflect: which labels does the baseline confuse most?
print("\\n── Baseline confusion pairs ──")
for true in LABELS:
    for pred in LABELS:
        if true != pred:
            n = sum(1 for t, p in zip(true_labels, groq_preds) if t == true and p == pred)
            if n > 0:
                print(f"  {true} predicted as {pred}: {n} case(s)")
"""),

cell("""\
from google.colab import files
files.download("baseline_results.json")
print("Downloaded baseline_results.json — commit this to your repo.")
"""),

# ─────────────────────────────────────────────────────────────────────────────
cell("# Section 6: Evaluation Metrics (Milestone 5)", "markdown"),

cell("""\
# ── Run after Section 3 (fine-tuning) is complete ────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Fine-tuned model predictions
ft_output    = trainer.predict(test_ds)
ft_pred_ids  = np.argmax(ft_output.predictions, axis=1)
ft_preds     = [ID2LABEL[i] for i in ft_pred_ids]

assert len(ft_preds) == len(true_labels), "Prediction count mismatch"
assert len(groq_preds) == len(true_labels), "Groq prediction count mismatch"

df_test_results["ft_pred"] = ft_preds

# Fine-tuned metrics
ft_acc      = accuracy_score(true_labels, ft_preds)
ft_macro_f1 = f1_score(true_labels, ft_preds, average="macro", labels=LABELS)
ft_report   = classification_report(true_labels, ft_preds, labels=LABELS, output_dict=True)
groq_report = classification_report(true_labels, groq_preds, labels=LABELS, output_dict=True)

print("=" * 60)
print("FINAL EVALUATION")
print("=" * 60)
print(f"\\nFine-Tuned DistilBERT:")
print(f"  Accuracy : {ft_acc:.4f}  ({ft_acc*100:.1f}%)")
print(f"  Macro F1 : {ft_macro_f1:.4f}")
print(classification_report(true_labels, ft_preds, labels=LABELS))

print(f"\\nGroq Zero-Shot Baseline:")
print(f"  Accuracy : {groq_acc:.4f}  ({groq_acc*100:.1f}%)")
print(f"  Macro F1 : {groq_macro_f1:.4f}")
print(classification_report(true_labels, groq_preds, labels=LABELS))

gap = ft_macro_f1 - groq_macro_f1
print(f"\\nMacro F1 gap (fine-tuned vs baseline): {gap:+.4f}")
if gap >= 0.05:
    print("Fine-tuning improved over baseline by >=5pp  SUCCESS")
elif gap >= 0:
    print("Fine-tuning improved but by <5pp — investigate")
else:
    print("WARNING: Fine-tuned underperforms zero-shot — check training")
"""),

cell("""\
# Confusion matrix
cm = confusion_matrix(true_labels, ft_preds, labels=LABELS)
assert cm.shape == (3, 3), f"Wrong shape: {cm.shape}"
assert cm.sum() == len(true_labels)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABELS, yticklabels=LABELS, ax=ax)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True",      fontsize=12)
ax.set_title("Fine-Tuned DistilBERT — Confusion Matrix", fontsize=13)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved confusion_matrix.png")
"""),

cell("""\
# Save combined evaluation_results.json
evaluation_results = {
    "finetuned_distilbert": {
        "accuracy":  ft_acc,
        "macro_f1":  ft_macro_f1,
        "per_class": ft_report,
    },
    "groq_zero_shot_llama3_70b": {
        "accuracy":   groq_acc,
        "macro_f1":   groq_macro_f1,
        "per_class":  groq_report,
        "parse_failures": len(failures),
    },
}
with open("evaluation_results.json", "w") as f:
    json.dump(evaluation_results, f, indent=2)
print("Saved evaluation_results.json")

# Print wrong predictions for README analysis
ft_errors = df_test_results[df_test_results["label"] != df_test_results["ft_pred"]]
print(f"\\nErrors: {len(ft_errors)}/{len(df_test_results)}")
print("\\n── Confusion pairs ──")
for true in LABELS:
    for pred in LABELS:
        if true != pred:
            n = len(ft_errors[(ft_errors["label"]==true) & (ft_errors["ft_pred"]==pred)])
            if n > 0:
                print(f"  {true} -> {pred}: {n}")

print("\\n── First 5 errors ──")
for _, row in ft_errors.head(5).iterrows():
    print(f"\\nTrue={row['label']} | Pred={row['ft_pred']}")
    print(f"Text: {row['text'][:250]}...")
    print("-"*40)
"""),

cell("""\
from google.colab import files
files.download("evaluation_results.json")
files.download("confusion_matrix.png")
print("Downloaded evaluation files. Commit them to your repo.")
"""),

# ─────────────────────────────────────────────────────────────────────────────
cell("# Section 7: Gradio Interface (Stretch)", "markdown"),

cell("""\
import gradio as gr
from transformers import pipeline as hf_pipeline

LABEL_DESCRIPTIONS = {
    "NTA": "Not the Asshole — your behavior was justified given the situation",
    "YTA": "You're the Asshole — you caused unjustified harm or acted unreasonably",
    "ESH": "Everyone Sucks Here — multiple parties behaved poorly; no clear winner",
}

verdict_clf = hf_pipeline(
    "text-classification",
    model="./best_model",
    tokenizer="./best_model",
    return_all_scores=True,
)

def predict_verdict(post_text):
    if not post_text or not post_text.strip():
        return "Please enter a post.", ""
    results   = verdict_clf(post_text[:2000])[0]
    predicted = max(results, key=lambda x: x["score"])
    label     = predicted["label"]
    confidence = predicted["score"]
    verdict_line = f"**{label}** ({confidence:.1%} confidence)\\n\\n{LABEL_DESCRIPTIONS[label]}"
    scores_lines = "\\n".join([
        f"{r['label']}: {r['score']:.1%}"
        for r in sorted(results, key=lambda x: -x["score"])
    ])
    return verdict_line, scores_lines

demo = gr.Interface(
    fn=predict_verdict,
    inputs=gr.Textbox(lines=8, placeholder="Paste your AITA post here...", label="Your Post"),
    outputs=[
        gr.Markdown(label="Verdict"),
        gr.Textbox(label="Confidence per Label", lines=3),
    ],
    title="TakeMeter — AITA Verdict Classifier",
    description=(
        "Paste an r/AmItheAsshole post and get a predicted verdict: "
        "NTA (Not the Asshole), YTA (You're the Asshole), or ESH (Everyone Sucks Here)."
    ),
    examples=[
        ["My roommate ate my labeled food for the third time. I locked my fridge. He's furious. AITA?"],
        ["I told my sister her baby was ugly at her baby shower. She asked me to leave. AITA?"],
        ["My neighbor plays loud music. I blasted music at 6am to retaliate. Now we're in a war. AITA?"],
    ],
)

demo.launch(share=True)
print("Gradio interface launched.")
"""),

]

# ─── Build .ipynb ──────────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
        "colab": {"gpuType": "T4"},
    },
    "cells": cells,
}

with open("takemeter_notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Generated takemeter_notebook.ipynb")
print("Upload this file to Colab: File -> Upload notebook")
