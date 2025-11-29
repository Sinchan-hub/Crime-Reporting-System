# train_ml.py
# Train sentence-transformer embeddings + classifier for crime type prediction
import os, pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Optional: import your preprocess_text if you placed it in a module
# from app import preprocess_text   # only if safe to import (circular import risk)
import re

def preprocess_text_local(text):
    # Minimal preprocessing aligned with your app's preprocess_text.
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s\u0900-\u097F\u0980-\u09FF\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ===== Sample dataset (expand this file with your real labelled data later) =====
# This is intentionally broad/multilingual to bootstrap performance.
texts = [
    # Theft / Robbery
    "phone was stolen", "my mobile got stolen", "wallet stolen in bus", "someone robbed me near market",
    "chain snatching incident", "my purse was stolen", "bike was stolen at night",
    "mera mobile chori ho gaya", "phone chori ho gayi", "pickpocket took my wallet",
    # Assault
    "i was assaulted", "someone hit me", "they beat me up", "man attacked me with stick",
    "maar diya mujhe", "dhakka mukki hui", "group attacked me",
    # Sexual assault
    "sexually assaulted in metro", "rape attempt reported", "molested by man", "bad touch by stranger",
    "balatkar hua", "chhedchhad hui", "zabardasti kiya",
    # Cybercrime
    "phishing email scam", "credit card fraud online", "upi fraud", "whatsapp scam link",
    "my bank account was hacked", "facebook hack",
    # Vandalism
    "car was damaged", "shop window broken", "property vandalized",
    # Kidnapping
    "kidnap attempt", "child missing suspected kidnapping", "attempt to abduct girl",
    # Harassment / Stalking
    "someone is following me", "stalking incident", "man threatened me", "galiyan di ja rahi hain"
]

labels = [
    # Theft / Robbery
    "Theft","Theft","Theft","Robbery","Robbery","Theft","Theft","Theft","Theft","Theft",
    # Assault
    "Assault","Assault","Assault","Assault","Assault","Assault","Assault",
    # Sexual assault
    "Sexual Assault","Sexual Assault","Sexual Assault","Sexual Assault","Sexual Assault","Sexual Assault","Sexual Assault",
    # Cybercrime
    "Cybercrime","Cybercrime","Cybercrime","Cybercrime","Cybercrime","Cybercrime",
    # Vandalism
    "Vandalism","Vandalism","Vandalism",
    # Kidnapping
    "Kidnapping","Kidnapping","Kidnapping",
    # Harassment / Stalking
    "Harassment","Stalking","Harassment","Harassment"
]

# Basic checks
assert len(texts) == len(labels), "Dataset mismatch: texts and labels length must match"

# Preprocess
texts_clean = [preprocess_text_local(t) for t in texts]

# Load sentence-transformer model
print("Loading sentence-transformer model (all-MiniLM-L6-v2). This will download the model once.")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Compute embeddings
embeddings = model.encode(texts_clean, show_progress_bar=True, convert_to_numpy=True)

# Label encode
le = LabelEncoder()
y = le.fit_transform(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.18, random_state=42)


# Train classifier
clf = LogisticRegression(max_iter=2000, solver='lbfgs', multi_class='multinomial')
clf.fit(X_train, y_train)

# Evaluate
pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Validation accuracy:", acc)
print(classification_report(y_test, pred, labels=np.unique(y_test), target_names=le.classes_))

# Save artifacts
os.makedirs('data/models', exist_ok=True)
joblib.dump(clf, 'data/models/clf.joblib')
joblib.dump(le, 'data/models/label_encoder.joblib')
# we don't need to save sentence-transformer (it will be loaded by name in app)
print("Saved classifier -> data/models/clf.joblib")
print("Saved label encoder -> data/models/label_encoder.joblib")
print("Training complete.")
