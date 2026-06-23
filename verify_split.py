"""
Verifies the train/val/test split and tokenization logic locally.
Mirrors what Section 1 and Section 2 of the Colab notebook will do.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
LABEL2ID = {"NTA": 0, "YTA": 1, "ESH": 2}
ID2LABEL  = {0: "NTA", 1: "YTA", 2: "ESH"}

# ── Section 1: Load ───────────────────────────────────────────────────────────
df = pd.read_csv("data/aita_labeled.csv")
print(f"Loaded {len(df)} examples. Columns: {list(df.columns)}")
assert df["text"].isna().sum() == 0
assert df["label"].isna().sum() == 0
assert set(df["label"].unique()) == {"NTA", "YTA", "ESH"}

df["label_id"] = df["label"].map(LABEL2ID)

print("\nLabel distribution:")
print(df["label"].value_counts().to_string())

# ── Section 2: Split ──────────────────────────────────────────────────────────
df_trainval, df_test = train_test_split(
    df, test_size=0.15, random_state=SEED, stratify=df["label_id"]
)
df_train, df_val = train_test_split(
    df_trainval, test_size=0.15/0.85, random_state=SEED, stratify=df_trainval["label_id"]
)

print(f"\nSplit sizes: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
print(f"Total: {len(df_train)+len(df_val)+len(df_test)}")

# Verify no overlap
train_ids = set(df_train["id"].tolist())
val_ids   = set(df_val["id"].tolist())
test_ids  = set(df_test["id"].tolist())
assert len(train_ids & val_ids)  == 0, "LEAK: train/val"
assert len(train_ids & test_ids) == 0, "LEAK: train/test"
assert len(val_ids   & test_ids) == 0, "LEAK: val/test"
print("No overlap between splits: PASSED")

# Verify all 3 classes in each split
for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
    labels = set(split_df["label"].unique())
    assert labels == {"NTA","YTA","ESH"}, f"{name} missing labels: {labels}"

print("\nTrain distribution:")
print(df_train["label"].value_counts().to_string())
print("\nVal distribution:")
print(df_val["label"].value_counts().to_string())
print("\nTest distribution:")
print(df_test["label"].value_counts().to_string())

# Save test set so baseline.py can reuse it (same split, same seed)
df_test.to_csv("data/test_set.csv", index=False)
print(f"\nSaved {len(df_test)} test examples to data/test_set.csv")
print("Section 1 + 2 verification PASSED.")
