# train_bert_crime.py
import os, json
import pandas as pd
from datasets import Dataset, ClassLabel
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import torch
from sklearn.model_selection import train_test_split
from collections import Counter

# Config
DATA_CSV = "crime_dataset_500.csv"   # place in same folder
OUTPUT_DIR = "ml_model/bert"
MODEL_NAME = "distilbert-base-uncased"
NUM_EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
MAX_LEN = 128
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_CSV)

# Ensure columns exist
df['description'] = df['description'].fillna('').astype(str)
df['text'] = df['description']  # Using only description column
df = df.dropna(subset=['text','crime_type']).reset_index(drop=True)


# Labels
labels = sorted(df['crime_type'].unique().tolist())
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}
df['label'] = df['crime_type'].map(label2id)

# Train/val split
train_df, val_df = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df['label'])

# Convert to huggingface Dataset
train_ds = Dataset.from_pandas(train_df[['text','label']])
val_ds = Dataset.from_pandas(val_df[['text','label']])

# Tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id
)

def preprocess(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=MAX_LEN)

train_ds = train_ds.map(preprocess, batched=True)
val_ds = val_ds.map(preprocess, batched=True)
train_ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])
val_ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True
)


def compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average='macro')
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save label map
with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
    json.dump({str(k): v for k,v in id2label.items()}, f)

print("Training complete. Saved to", OUTPUT_DIR)
