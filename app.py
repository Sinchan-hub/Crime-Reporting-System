# ============================================================
# PART 1 — IMPORTS + GLOBAL CONFIG + UTILITIES
# ============================================================

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

import os, csv, uuid, datetime, math, pickle, shutil, io, base64, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime, timedelta
# ------------------- BEGIN ML INTEGRATION BLOCK A -------------------
# Add this after your imports (top of file). It loads ML models + helpers.

import os
import joblib
import pickle
import math

# Optional: spaCy for NER
try:
    import spacy
    _spacy_available = True
except Exception:
    _spacy_available = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import json
import os

# Base ml_model folder (adjust if your structure is different)
ML_BASE = os.path.join(os.path.dirname(__file__), 'ml_model')

# classifier (existing)
CLASSIFIER_DIR = os.path.join(ML_BASE, 'classifier')
CLASSIFIER_MODEL = os.path.join(CLASSIFIER_DIR, 'model.pkl')
CLASSIFIER_VECT = os.path.join(CLASSIFIER_DIR, 'vectorizer.pkl')
CLASSIFIER_LE = os.path.join(CLASSIFIER_DIR, 'label_encoder.pkl')

# severity
SEV_DIR = os.path.join(ML_BASE, 'severity')
SEV_MODEL = os.path.join(SEV_DIR, 'sev_model.pkl')
SEV_VECT = os.path.join(SEV_DIR, 'sev_vectorizer.pkl')
SEV_LE = os.path.join(SEV_DIR, 'sev_label_encoder.pkl')

# anomaly
ANOMALY_DIR = os.path.join(ML_BASE, 'anomaly')
ISO_FOREST_FILE = os.path.join(ANOMALY_DIR, 'iso_forest.pkl')

# forecast
FORECAST_DIR = os.path.join(ML_BASE, 'forecast')
ARIMA_FILE = os.path.join(FORECAST_DIR, 'arima_model.pkl')

# NER model name to try to load if spaCy available
_SPACY_MODEL_NAME = "en_core_web_sm"

# Global holders
_ml_classifier = None
_ml_clf_vect = None
_ml_clf_le = None

_sev_model = None
_sev_vect = None
_sev_le = None

_iso_forest = None
_forecast = None
_spacy_nlp = None

def _try_load_pickle(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        try:
            with open(path,'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

def load_ml_models():
    global _ml_classifier, _ml_clf_vect, _ml_clf_le
    global _sev_model, _sev_vect, _sev_le
    global _iso_forest, _forecast, _spacy_nlp

    # Classifier (existing)
    if os.path.exists(CLASSIFIER_MODEL):
        try:
            _ml_classifier = _try_load_pickle(CLASSIFIER_MODEL)
            _ml_clf_vect = _try_load_pickle(CLASSIFIER_VECT)
            _ml_clf_le = _try_load_pickle(CLASSIFIER_LE)
            print("Loaded classifier model from", CLASSIFIER_DIR)
        except Exception as e:
            print("Classifier load failed:", e)

    # Severity
    if os.path.exists(SEV_MODEL):
        _sev_model = _try_load_pickle(SEV_MODEL)
        _sev_vect = _try_load_pickle(SEV_VECT)
        _sev_le = _try_load_pickle(SEV_LE)
        if _sev_model:
            print("Severity model loaded from", SEV_DIR)

    # Anomaly
    if os.path.exists(ISO_FOREST_FILE):
        _iso_forest = _try_load_pickle(ISO_FOREST_FILE)
        if _iso_forest is not None:
            print("IsolationForest anomaly model loaded.")

    # Forecast (placeholder)
    if os.path.exists(ARIMA_FILE):
        _forecast = _try_load_pickle(ARIMA_FILE)
        print("Forecast (placeholder) loaded.")

    # spaCy NER
    if _spacy_available:
        try:
            _spacy_nlp = spacy.load(_SPACY_MODEL_NAME)
            print("spaCy model", _SPACY_MODEL_NAME, "loaded.")
        except Exception:
            try:
                # try to download automatically if not present (best-effort)
                from spacy.cli import download as spacy_download
                spacy_download(_SPACY_MODEL_NAME)
                _spacy_nlp = spacy.load(_SPACY_MODEL_NAME)
                print("spaCy model downloaded and loaded.")
            except Exception:
                _spacy_nlp = None
                print("spaCy NER not available.")
# ---- BERT MODEL LOADING ----
BERT_DIR = os.path.join(os.path.dirname(__file__), "ml_model", "bert")

tokenizer_bert = None
bert_model = None
bert_label_map = None

if os.path.isdir(BERT_DIR):
    try:
        tokenizer_bert = DistilBertTokenizerFast.from_pretrained(BERT_DIR)
        bert_model = DistilBertForSequenceClassification.from_pretrained(BERT_DIR)
        with open(os.path.join(BERT_DIR,"label_map.json"), "r") as f:
            bert_label_map = json.load(f)
        print("BERT Loaded Successfully")
    except Exception as e:
        print("BERT load failed:", e)
# ---- BERT PREDICTION FUNCTION ----
def predict_crime_type_bert(text):
    if not tokenizer_bert or not bert_model:
        return None

    try:
        inputs = tokenizer_bert(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        if torch.cuda.is_available():
            bert_model.to("cuda")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = bert_model(**inputs)
            logits = outputs.logits
            pred_idx = torch.argmax(logits, dim=1).cpu().item()

        return bert_label_map.get(str(pred_idx), None)

    except Exception as e:
        print("BERT prediction error:", e)
        return None

# Call once at startup
try:
    load_ml_models()
except Exception as e:
    print("ML load error:", e)


# ---------- Helper functions ----------
def predict_crime_type(text):
    """Return predicted crime label (string) or None"""
    if not text:
        return None
    try:
        if _ml_classifier and _ml_clf_vect:
            X = _ml_clf_vect.transform([text])
            pred_idx = _ml_classifier.predict(X)
            if _ml_clf_le:
                try:
                    return _ml_clf_le.inverse_transform(pred_idx)[0]
                except Exception:
                    return str(pred_idx[0])
            return str(pred_idx[0])
    except Exception as e:
        print("Crime type predict failed:", e)
    return None

def predict_severity(text):
    """Return severity label (string) or None"""
    if not text:
        return None
    try:
        if _sev_model and _sev_vect:
            X = _sev_vect.transform([text])
            p = _sev_model.predict(X)
            if _sev_le:
                try:
                    return _sev_le.inverse_transform(p)[0]
                except Exception:
                    return str(p[0])
            return str(p[0])
    except Exception as e:
        print("Severity predict failed:", e)
    return None

def extract_location_with_ner(text):
    """Return first GPE/LOC entity if spaCy available; otherwise None"""
    if not text or _spacy_nlp is None:
        return None
    try:
        doc = _spacy_nlp(text)
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC", "FAC", "ORG"):
                return ent.text
    except Exception as e:
        print("NER extraction failed:", e)
    return None

def anomaly_score_for_point(value):
    """Return anomaly flag True/False. If iso forest unavailable, return False"""
    if _iso_forest is None:
        return False
    try:
        import numpy as np
        arr = np.array([[float(value)]])
        score = _iso_forest.predict(arr)  # returns 1 for normal, -1 for anomaly
        return int(score[0]) == -1
    except Exception as e:
        print("Anomaly test failed:", e)
        return False

# ------------------- END ML INTEGRATION BLOCK A -------------------



# >>> news integration imports (paste after your datetime import)
import feedparser
import requests
import news_fetcher   # new file - import the module we just created
from apscheduler.schedulers.background import BackgroundScheduler
# <<< end news imports
# ---------- Utilities ----------
import csv
# other imports already present...

def read_csv(path):
    """
    Safe CSV loader that returns list of dicts.
    Place this near the top of app.py (after imports).
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


# Try SBERT
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except:
    SBERT_AVAILABLE = False

# Twilio optional
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except:
    TWILIO_AVAILABLE = False


# ============================================================
# PATHS + FOLDERS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
DATA_FOLDER = os.path.join(BASE_DIR, "data")
MODEL_FOLDER = os.path.join(DATA_FOLDER, "models")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

USERS_CSV = os.path.join(DATA_FOLDER, "users.csv")
REPORTS_CSV = os.path.join(DATA_FOLDER, "reports.csv")
NOTIF_CSV = os.path.join(DATA_FOLDER, "notifications.csv")
POLICE_CSV = os.path.join(DATA_FOLDER, "police_stations.csv")
ACTIVE_SOS_CSV = os.path.join(DATA_FOLDER, "active_sos.csv")
EMERGENCY_CSV = os.path.join(DATA_FOLDER, "emergency_contacts.csv")
SMS_LOG = os.path.join(DATA_FOLDER, "sms_log.txt")

TEXT_MODEL_TFIDF = os.path.join(MODEL_FOLDER, "crime_model_tfidf.pkl")
VECTORIZER_PATH = os.path.join(MODEL_FOLDER, "vectorizer.pkl")
SBERT_MODEL_PATH = os.path.join(MODEL_FOLDER, "sbert_model.pkl")


# REPORT CSV HEADER
REPORTS_HEADER = [
    "id","name","title","description","image","lat","lon",
    "location_name","incident_date","status","assigned_police",
    "predicted_type","severity","anomaly_flag","created_at"
]


ALLOWED_EXT = {'png','jpg','jpeg','gif'}
MAX_CONTENT = 16 * 1024 * 1024


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def allowed(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

def ensure_csv(path, header):
    if not os.path.exists(path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f); writer.writerow(header)

ensure_csv(USERS_CSV, ["id","role","name","email","phone","password","created_at"])
ensure_csv(REPORTS_CSV, REPORTS_HEADER)
ensure_csv(NOTIF_CSV, ["id","report_id","police_name","message","created_at","is_sos","seen"])
ensure_csv(POLICE_CSV, ["station_name","email","phone","lat","lon"])
ensure_csv(ACTIVE_SOS_CSV, ["id","name","lat","lon","created_at","status"])
ensure_csv(EMERGENCY_CSV, ["name","contact_phone"])


# ============================================================
# SEED POLICE STATIONS
# ============================================================
def seed_police_stations():
    with open(POLICE_CSV, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        stations = [
            ("MG Road Police Station","mgroad@police.in","08025555555","12.9750","77.6033"),
            ("Koramangala Police Station","koramangala@police.in","08026666666","12.9352","77.6245"),
            ("Indiranagar Police Station","indiranagar@police.in","08024444444","12.9692","77.6411"),
            ("Whitefield Police Station","whitefield@police.in","08027777777","12.9698","77.7499"),
            ("Jayanagar Police Station","jayanagar@police.in","08023333333","12.9250","77.5938")
        ]
        with open(POLICE_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for s in stations:
                writer.writerow(s)

seed_police_stations()

# ============================================================
# FLASK INIT
# ============================================================
app = Flask(__name__)
app.secret_key = "your-secret-key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.permanent_session_lifetime = timedelta(days=7)

# END OF PART 1
# ============================================================
# PART 2 — TEXT PREPROCESS + ML MODELS + KMEANS + GEO HELPERS
# ============================================================

def preprocess_text(text):
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\u0900-\u097F]", " ", text)
    replacements = {
        "chori":"theft", "phone stolen":"theft",
        "wallet stolen":"theft", "loot":"robbery",
        "snatch":"robbery", "molest":"sexual assault",
        "rape":"rape", "fraud":"cybercrime", "scam":"cybercrime",
        "beat me":"assault", "attacked":"assault"
    }
    for k,v in replacements.items():
        text = text.replace(k, v)
    text = re.sub(r"\s+"," ", text)
    return text


# ============================================================
# TF-IDF TRAINER (fallback)
# ============================================================
def train_tfidf_model():
    texts = [
        "phone stolen", "wallet stolen", "bike stolen",
        "I was assaulted", "beat me", "attacked with stick",
        "rape attempt", "molested", "sexual assault",
        "upi fraud", "scam email", "hacked account",
        "car vandalized", "kidnap attempt"
    ]
    labels = [
        "Theft","Theft","Theft",
        "Assault","Assault","Assault",
        "Sexual Assault","Sexual Assault","Sexual Assault",
        "Cybercrime","Cybercrime","Cybercrime",
        "Vandalism","Kidnapping"
    ]
    vect = TfidfVectorizer(max_features=1500)
    X = vect.fit_transform(texts)
    clf = RandomForestClassifier(n_estimators=150)
    clf.fit(X, labels)

    with open(TEXT_MODEL_TFIDF,'wb') as f: pickle.dump(clf,f)
    with open(VECTORIZER_PATH,'wb') as f: pickle.dump(vect,f)
    return clf, vect


# ============================================================
# SBERT (if available) or fallback to TF-IDF
# ============================================================
def load_models():
    if SBERT_AVAILABLE:
        try:
            with open(SBERT_MODEL_PATH,'rb') as f: clf = pickle.load(f)
            with open(os.path.join(MODEL_FOLDER,"sbert_encoder.pkl"),'rb') as f: enc = pickle.load(f)
            return "sbert", clf, enc
        except:
            pass

    # TF-IDF fallback
    if os.path.exists(TEXT_MODEL_TFIDF):
        with open(TEXT_MODEL_TFIDF,'rb') as f: clf = pickle.load(f)
        with open(VECTORIZER_PATH,'rb') as f: vect = pickle.load(f)
        return "tfidf", clf, vect

    clf, vect = train_tfidf_model()
    return "tfidf", clf, vect


MODEL_MODE, text_clf, text_encoder = load_models()


# ============================================================
# HOTSPOT KMEANS
# ============================================================
def train_kmeans(k=3):
    try:
        df = pd.read_csv(REPORTS_CSV)
        df = df[(df['lat']!='') & (df['lon']!='')]
        coords = df[['lat','lon']].astype(float)
        if len(coords) >= k:
            model = KMeans(n_clusters=k)
            model.fit(coords)
            with open(KMEANS_MODEL,'wb') as f: pickle.dump(model,f)
            return model
    except:
        pass
    return None


def load_kmeans():
    if os.path.exists(KMEANS_MODEL):
        try:
            with open(KMEANS_MODEL,'rb') as f:
                return pickle.load(f)
        except:
            pass
    return None


# ============================================================
# GEO HELPERS
# ============================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def assign_nearest_police(lat, lon):
    try:
        lat = float(lat)
        lon = float(lon)
    except:
        with open(POLICE_CSV,'r') as f:
            r = list(csv.DictReader(f))
            return r[0]['station_name'] if r else "Central Police"

    stations = list(csv.DictReader(open(POLICE_CSV,'r')))
    best = None
    bestd = 1e9

    for s in stations:
        try:
            d = haversine(lat, lon, float(s['lat']), float(s['lon']))
            if d < bestd:
                bestd = d
                best = s
        except:
            continue

    return best['station_name'] if best else "Central Police"


# ============================================================
# CRIME PREDICTION WRAPPER
# ============================================================
def predict_crime(text):
    clean = preprocess_text(text)

    if MODEL_MODE == "sbert":
        try:
            emb = text_encoder.encode([clean])
            return text_clf.predict(emb)[0]
        except:
            pass

    try:
        X = text_encoder.transform([clean])
        return text_clf.predict(X)[0]
    except:
        return "Unknown"
    
def read_csv(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    except:
        return []

def notify_emergency_contacts(user_id, message):
    """
    Reads emergency_contacts.csv and logs or sends SMS.
    Since Twilio may or may not be configured, this function
    will safely log instead of crashing.
    """

    contacts = []
    try:
        with open(EMERGENCY_CSV, 'r', encoding='utf-8') as f:
            contacts = list(csv.DictReader(f))
    except Exception as e:
        print("Failed to read emergency contacts:", e)
        return

    # Loop through contacts
    for c in contacts:
        phone = c.get("contact_phone", "")
        if not phone:
            continue

        log_entry = f"[EMERGENCY_SMS] To: {phone} | Msg: {message}\n"
        print(log_entry)

        # If Twilio is configured, send SMS
        if TWILIO_AVAILABLE:
            try:
                account_sid = "your_account_sid"
                auth_token = "your_auth_token"
                client = TwilioClient(account_sid, auth_token)

                client.messages.create(
                    body=message,
                    from_="+1234567890",  # your Twilio phone
                    to=phone
                )
            except Exception as e:
                print("Twilio SMS send failed:", e)

        # Log SMS into sms_log.txt
        try:
            with open(SMS_LOG, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except:
            pass

# ------------------- BEGIN ML INTEGRATION BLOCK B -------------------
# Insert this inside your report_new route, just BEFORE the code that writes the report to CSV/db.

# assume `report` is the dict that will be saved; if your code names it differently,
# set `report_obj` to that variable name.

# try:
#     # unify variable name to 'report' for this snippet (adjust if needed)
#     # If your code uses some other var name like `row` or `new_report`, change below accordingly.
#     report_obj = report  # <- if your variable is named differently, replace 'report' with it

#     # 1) Predict crime type if missing / or update predicted_type column
#     # prefer using the title+description text
#     text_for_pred = " ".join(filter(None, [report_obj.get('title',''), report_obj.get('description','')]))
#     predicted = predict_crime_type(text_for_pred)
#     if predicted:
#         # add predicted_type field to the record so it persists in CSV / DB
#         report_obj['predicted_type'] = predicted

#     # 2) Predict severity
#     sev = predict_severity(text_for_pred)
#     if sev:
#         report_obj['severity'] = sev

#     # 3) Try to extract location from description/title using NER (best-effort)
#     ner_loc = None
#     try:
#         ner_loc = extract_location_with_ner(report_obj.get('description','') or report_obj.get('title',''))
#     except Exception:
#         ner_loc = None
#     if ner_loc and not report_obj.get('location_name'):
#         # if the report lacks explicit location_name, fill it
#         report_obj['location_name_from_ner'] = ner_loc

#     # 4) Basic anomaly check: if there is an explicit numerical 'lat' or 'lon' or 'some metric' we can test.
#     # Here we do a trivial check: if lat exists use it as numeric input for iso forest.
#     try:
#         lat_val = report_obj.get('lat') or report_obj.get('latitude') or None
#         if lat_val:
#             # try to convert to float (only for anomaly model test)
#             try:
#                 val = float(lat_val)
#                 is_anom = anomaly_score_for_point(val)
#                 report_obj['anomaly_flag'] = '1' if is_anom else '0'
#             except Exception:
#                 report_obj['anomaly_flag'] = '0'
#         else:
#             report_obj['anomaly_flag'] = '0'
#     except Exception:
#         report_obj['anomaly_flag'] = '0'

#     # End of ML enrichment. Now your existing save logic should run unchanged and persist these fields.
# except Exception as e:
#     # defensive: don't break the report flow if ML fails
#     print("ML enrichment inside report_new failed:", e)
# ------------------- END ML INTEGRATION BLOCK B -------------------

# ============================================================
# PART 3 — ROUTES: auth, account, dashboards, reports, APIs
# Paste THIS after Part 2
# ============================================================

@app.before_request
def make_session_permanent():
    session.permanent = True

@app.route('/')
def index():
    bg = None
    bg_path = os.path.join(app.config['UPLOAD_FOLDER'], 'background.jpg')
    if os.path.exists(bg_path):
        bg = '/static/uploads/background.jpg'
    return render_template('index.html', user=session.get('user'), bg=bg)


# ----------------- AUTH / REGISTER / LOGIN / LOGOUT -----------------
@app.route('/register/user', methods=['GET','POST'])
def register_user():
    if request.method == 'POST':
        name = (request.form.get('name') or '').strip()
        email = (request.form.get('email') or '').strip()
        phone = (request.form.get('phone') or '').strip()
        password = (request.form.get('password') or '').strip()
        if not (name and email and phone and password):
            flash('Please fill all fields','warning')
            return redirect(request.url)
        hashed = generate_password_hash(password)
        uid = str(uuid.uuid4())[:8]
        with open(USERS_CSV,'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([uid, 'user', name, email, phone, hashed, datetime.utcnow().isoformat()])
        flash('User registered — please login.','success')
        return redirect(url_for('login_user'))
    return render_template('register_user.html')


@app.route('/register/police', methods=['GET','POST'])
def register_police():
    if request.method == 'POST':
        name = (request.form.get('name') or '').strip()
        email = (request.form.get('email') or '').strip()
        phone = (request.form.get('phone') or '').strip()
        password = (request.form.get('password') or '').strip()
        if not (name and email and phone and password):
            flash('Please fill all fields','warning'); return redirect(request.url)
        hashed = generate_password_hash(password)
        uid = str(uuid.uuid4())[:8]
        with open(USERS_CSV,'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([uid, 'police', name, email, phone, hashed, datetime.utcnow().isoformat()])
        flash('Police registered — please login.','success')
        return redirect(url_for('login_police'))
    return render_template('register_police.html')


@app.route('/login/user', methods=['GET','POST'])
def login_user():
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip()
        password = (request.form.get('password') or '').strip()
        try:
            with open(USERS_CSV, 'r', encoding='utf-8') as f:
                users = list(csv.DictReader(f))
        except:
            users = []
        for u in users:
            if u.get('email') == email and u.get('role') == 'user':
                try:
                    if check_password_hash(u.get('password',''), password):
                        session['user'] = {'id':u.get('id'),'role':u.get('role'),'name':u.get('name'),'email':u.get('email'),'phone':u.get('phone')}
                        flash('Login successful','success')
                        return redirect(url_for('index'))
                except Exception:
                    pass
        flash('Invalid credentials','danger')
    return render_template('login_user.html')


@app.route('/login/police', methods=['GET','POST'])
def login_police():
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip()
        password = (request.form.get('password') or '').strip()
        try:
            with open(USERS_CSV, 'r', encoding='utf-8') as f:
                users = list(csv.DictReader(f))
        except:
            users = []
        for u in users:
            if u.get('email') == email and u.get('role') == 'police':
                try:
                    if check_password_hash(u.get('password',''), password):
                        session['user'] = {'id':u.get('id'),'role':u.get('role'),'name':u.get('name'),'email':u.get('email'),'phone':u.get('phone')}
                        flash('Login successful','success')
                        return redirect(url_for('index'))
                except Exception:
                    pass
        flash('Invalid credentials','danger')
    return render_template('login_police.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out','info')
    return redirect(url_for('index'))


# ----------------- ACCOUNT -----------------
@app.route('/account')
def account():
    if 'user' not in session:
        flash('Login required','warning'); return redirect(url_for('index'))
    return render_template('account.html', user=session['user'])


@app.route('/account/change_password', methods=['GET','POST'])
def change_password():
    if 'user' not in session:
        flash('Login required','warning'); return redirect(url_for('index'))
    if request.method == 'POST':
        old = (request.form.get('old_password') or '')
        new = (request.form.get('new_password') or '')
        confirm = (request.form.get('confirm_password') or '')
        if not (old and new and confirm):
            flash('All fields required','warning'); return redirect(request.url)
        if new != confirm:
            flash('New passwords do not match','danger'); return redirect(request.url)
        uid = session['user']['id']
        rows = []
        with open(USERS_CSV,'r',encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        updated = False
        for r in rows:
            if r.get('id') == uid:
                if not check_password_hash(r.get('password',''), old):
                    flash('Old password incorrect','danger'); return redirect(request.url)
                r['password'] = generate_password_hash(new)
                updated = True
        if updated:
            with open(USERS_CSV,'w',newline='',encoding='utf-8') as f:
                writer = csv.writer(f); writer.writerow(['id','role','name','email','phone','password','created_at'])
                for r in rows:
                    writer.writerow([r.get('id',''), r.get('role',''), r.get('name',''), r.get('email',''), r.get('phone',''), r.get('password',''), r.get('created_at','')])
            flash('Password changed','success'); return redirect(url_for('account'))
    return render_template('change_password.html', user=session['user'])


# ----------------- USER DASHBOARD -----------------
@app.route('/user/dashboard')
def user_dashboard():
    if 'user' not in session or session['user'].get('role') != 'user':
        flash('Login required','warning')
        return redirect(url_for('login_user'))

    uid = session['user']['name']

    try:
        with open(REPORTS_CSV,'r',encoding='utf-8') as f:
            all_reports = list(csv.DictReader(f))
    except Exception:
        all_reports = []

    # Reports filed by this user
    my_reports = [r for r in all_reports if r.get('name') == uid]
    community_reports = all_reports
    total_reports = len(my_reports)

    # --- Universal normalizer ---
    import re
    def norm(v):
        if not v:
            return ""
        v = v.strip().lower()
        v = re.sub(r'[^a-z ]', '', v)   # removes \r, \n, unicode spaces etc.
        return v

    # --- Status Counting (bullet-proof) ---
    active_reports  = len([r for r in my_reports 
                           if norm(r.get('status')) == 'reported'])

    pending_reports = len([r for r in my_reports 
                           if norm(r.get('status')) in ('in progress', 'inprogress')])

    closed_reports  = len([r for r in my_reports 
                           if norm(r.get('status')) in ('closed', 'solved', 'resolved')])

    return render_template(
        'user_dashboard.html',
        user=session['user'],
        my_reports=my_reports,
        community_reports=community_reports,
        total_reports=total_reports,
        active_reports=active_reports,
        pending_reports=pending_reports,
        closed_reports=closed_reports
    )



@app.route('/police/dashboard')
def police_dashboard():
    if 'user' not in session or session['user'].get('role') != 'police':
        flash("Login required", "warning")
        return redirect(url_for('login_police'))

    # Load all reports
    reports = read_csv(REPORTS_CSV)
    notifs = read_csv(NOTIF_CSV)

    # Identify current police station
    station_name = "Central Police Station"
    try:
        stations = read_csv(POLICE_CSV)
        police_email = session['user'].get('email','').strip().lower()
        for s in stations:
            if s.get('email','').strip().lower() == police_email:
                station_name = (s.get('station_name') or station_name).strip().title()
                break
    except:
        pass

    # Load SOS alerts correctly
    sos_alerts = read_csv(ACTIVE_SOS_CSV)

    # Assigned reports
    def norm(v): return (v or "").strip().lower()
    assigned_reports = [r for r in reports if norm(r.get('assigned_police')) == norm(station_name)]

    # Stats
    total_reports = len(reports)
    assigned_count = len(assigned_reports)
    in_progress = len([r for r in assigned_reports if r.get("status","").lower() == "in progress"])
    closed_count = len([r for r in assigned_reports if r.get("status","").lower() == "closed"])

    return render_template(
        "police_dashboard.html",
        user=session['user'],
        reports=reports,
        assigned_reports=assigned_reports,
        notifs=notifs,
        total_reports=total_reports,
        assigned_count=assigned_count,
        in_progress_count=in_progress,
        closed_count=closed_count,
        station_name=station_name,
        sos_alerts=sos_alerts  # <-- THIS FIXES YOUR SOS MAP
    )


@app.route('/police/report/<report_id>', methods=['GET','POST'])
def police_view_report(report_id):
    if 'user' not in session or session['user'].get('role') != 'police':
        flash("Login required", "warning")
        return redirect(url_for('login_police'))

    reports = read_csv(REPORTS_CSV)

    # locate report
    report = None
    for r in reports:
        if r.get('id') == report_id:
            report = r
            break

    if not report:
        flash("Report not found", "danger")
        return redirect(url_for('police_dashboard'))

    if request.method == 'POST':
        new_status = request.form.get('status') or report.get('status')
        # update in-memory
        for r in reports:
            if r.get('id') == report_id:
                r['status'] = new_status
        # save back to CSV - REQUIRED: REPORTS_HEADER must be defined in your app.py
        with open(REPORTS_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(REPORTS_HEADER)
            for r in reports:
                writer.writerow([ r.get(col,'') for col in REPORTS_HEADER ])
        flash("Status updated", "success")
        return redirect(url_for('police_view_report', report_id=report_id))

    # render page
    return render_template("police_view_report.html", report=report, user=session['user'])



# ----------------- NEW REPORT -----------------
# ----------------- NEW REPORT -----------------
@app.route('/report/new', methods=['GET','POST'])
def report_new():
    if 'user' not in session or session['user'].get('role') != 'user':
        flash('Login required','warning'); return redirect(url_for('login_user'))

    if request.method == 'POST':
        title = (request.form.get('title') or '').strip()
        desc = (request.form.get('description') or '').strip()
        lat = (request.form.get('lat') or '').strip()
        lon = (request.form.get('lon') or '').strip()
        location_name = (request.form.get('location_name') or '').strip()
        incident_date = (request.form.get('incident_date') or '').strip()
        file = request.files.get('image')
        use_as_bg = request.form.get('use_as_bg')
        
        if not title or not desc:
            flash('Please provide title and description','warning')
            return redirect(request.url)

        imgname = ''
        if file and file.filename != '' and allowed(file.filename):
            fn = secure_filename(file.filename)
            unique = f"{uuid.uuid4().hex[:8]}_{fn}"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], unique))
            imgname = unique

            if use_as_bg:
                try:
                    shutil.copy(
                        os.path.join(app.config['UPLOAD_FOLDER'], unique),
                        os.path.join(app.config['UPLOAD_FOLDER'], 'background.jpg')
                    )
                except:
                    pass

        # OLD predicted type
        predicted = "Unknown"
        try:
            predicted = predict_crime(title + " " + desc)
        except:
            predicted = "Unknown"

        # -----------------------------------------------------------
        # 🚀 ML BLOCK B — NEW ML FEATURES (correctly implemented)
        # -----------------------------------------------------------

        text_for_ml = f"{title} {desc}".strip()

        # 1. NEW ML: Crime type (Logistic Regression)
        # 1. NEW ML: Crime type — BERT (preferred)
        bert_pred = predict_crime_type_bert(text_for_ml)
        if bert_pred:
            predicted = bert_pred
        else:
        # fallback to Logistic Regression
            try:
                new_predicted_type = predict_crime_type(text_for_ml)
                if new_predicted_type:
                    predicted = new_predicted_type
            except Exception as e:
                print("ML type error:", e)


        # 2. NEW ML: Severity
        severity = "medium"
        try:
            severity = predict_severity(text_for_ml) or "unknown"
        except Exception as e:
            print("severity error:", e)

        # 3. NEW ML: NER Location Extraction
        ner_location = None
        try:
            ner_location = extract_location_with_ner(desc)
        except:
            ner_location = None

        if not location_name and ner_location:
            location_name = ner_location

        # 4. NEW ML: Anomaly check (based on lat numeric)
        anomaly_flag = "0"
        try:
            if lat:
                anomaly_flag = "1" if anomaly_score_for_point(float(lat)) else "0"
        except:
            anomaly_flag = "0"

        # -----------------------------------------------------------
        # END ML BLOCK B
        # -----------------------------------------------------------

        assigned = assign_nearest_police(lat, lon)
        assigned = (assigned or "").strip().title()
        rid = str(uuid.uuid4())[:8]
        created = datetime.utcnow().isoformat()

        # UPDATED CSV ROW (added severity + anomaly_flag + ner_location)
        with open(REPORTS_CSV,'a',newline='',encoding='utf-8') as f:
            writer = csv.writer(f)
            row = [
                rid,
                session['user']['name'],
                title,
                desc,
                imgname,
                lat,
                lon,
                location_name,
                incident_date or '',
                'Reported',
                assigned,
                predicted,        # ML crime type
                severity,         # NEW ML severity
                anomaly_flag,     # NEW anomaly
                created
            ]
            writer.writerow(row)

        nid = str(uuid.uuid4())[:8]
        msg = f"New report {title} at {location_name or (lat+','+lon)}"

        with open(NOTIF_CSV,'a',newline='',encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([nid, rid, assigned, msg, created, "0", "False"])

        try:
            train_kmeans()
        except:
            pass

        flash(f'Crime reported successfully. Predicted type: {predicted} | Severity: {severity}', 'success')
        return redirect(url_for('user_dashboard'))

    return render_template('report_new.html', user=session.get('user'))



# ----------------- REPORT VIEW + NEARBY API -----------------
@app.route('/report/view/<rid>')
def report_view(rid):
    try:
        with open(REPORTS_CSV,'r',encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r.get('id') == rid:
                    return render_template('report_view.html', r=r, user=session.get('user'))
    except:
        pass
    flash('Report not found','warning'); return redirect(url_for('index'))


@app.route('/reports/nearby')
def reports_nearby():
    out = []
    try:
        with open(REPORTS_CSV,'r',encoding='utf-8') as f:
            for r in csv.DictReader(f):
                try:
                    out.append({
                        'id': r.get('id',''),
                        'title': r.get('title',''),
                        'lat': float(r.get('lat')) if r.get('lat') else None,
                        'lon': float(r.get('lon')) if r.get('lon') else None,
                        'status': r.get('status','Reported'),
                        'predicted_type': r.get('predicted_type','Unknown')
                    })
                except:
                    out.append({'id': r.get('id',''), 'title': r.get('title',''), 'lat': None, 'lon': None, 'status': r.get('status','Reported'), 'predicted_type': r.get('predicted_type','Unknown')})
    except:
        pass
    return jsonify(out)

# ===== News routes (insert above existing @app.route('/hotspots') ) =====
@app.route('/news')
def news_page():
    # shows the Inshorts-like feed + map
    return render_template('news.html', user=session.get('user'))

@app.route('/api/news')
def api_news():
    out = []
    try:
        with open(os.path.join(DATA_FOLDER, "news.csv"), 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                out.append({
                    "id": r.get("id",""),
                    "title": r.get("title",""),
                    "description": r.get("description",""),
                    "location_text": r.get("location_text",""),
                    "lat": float(r.get("lat")) if r.get("lat") else None,
                    "lon": float(r.get("lon")) if r.get("lon") else None,
                    "source": r.get("source",""),
                    "url": r.get("url",""),
                    "published_at": r.get("published_at",""),
                    "is_hotspot": True if r.get("is_hotspot","").lower()=="true" else False,
                    "hotspot_group": r.get("hotspot_group","")
                })
    except Exception:
        pass
    return jsonify(out)
# ===== end news routes =====

@app.route('/hotspots')
def hotspots():

    def valid_point(lat, lon):
        try:
            lat = float(lat)
            lon = float(lon)
            # Accept only India region bounding box
            if 6.0 <= lat <= 38.0 and 68.0 <= lon <= 98.0:
                return True
            return False
        except:
            return False

    centers = []

    # ----- USER REPORTS -----
    try:
        with open(REPORTS_CSV, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                lat = r.get("lat")
                lon = r.get("lon")
                if valid_point(lat, lon):
                    centers.append([float(lat), float(lon)])
    except:
        pass

    # ----- NEWS INCIDENTS -----
    try:
        NEWS_CSV = os.path.join(DATA_FOLDER, "news.csv")
        with open(NEWS_CSV, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                lat = r.get("lat")
                lon = r.get("lon")
                if valid_point(lat, lon):
                    centers.append([float(lat), float(lon)])
    except:
        pass

    # If empty fallback to Bangalore
    if not centers:
        return jsonify([[12.9716, 77.5946]])

    return jsonify(centers)




@app.route('/hotspots/map')
def hotspots_map():
    return render_template('hotspots_map.html')


# ----------------- SUMMARY PAGE (simple stats + small PNG chart) -----------------
@app.route('/summary')
def summary():
    try:
        df = pd.read_csv(REPORTS_CSV)
    except:
        df = pd.DataFrame(columns=['predicted_type','status'])
    total = len(df)
    active = len(df[df.get('status')=='Reported']) if 'status' in df.columns else total
    closed = len(df[df.get('status')=='Closed']) if 'status' in df.columns else 0
    by_type = df['predicted_type'].value_counts().to_dict() if 'predicted_type' in df.columns else {}

    # small PNG chart generated server-side for legacy clients
    labels = list(by_type.keys()) if by_type else ['No Data']
    values = list(by_type.values()) if by_type else [1]
    try:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(labels, values, color='#0b84ff')
        ax.set_title('Reports by Predicted Type')
        plt.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format='png'); buf.seek(0)
        chart_b64 = base64.b64encode(buf.read()).decode('utf-8'); buf.close()
    except Exception:
        chart_b64 = None

    return render_template('summary.html', total=total, active=active, closed=closed, by_type=by_type, chart_data=chart_b64, user=session.get('user'))


# ----------------- ERROR HANDLERS -----------------
@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', message=str(e)), 500

@app.errorhandler(413)
def file_too_large(e):
    flash('Uploaded file is too large (max 16MB)','danger'); return redirect(request.referrer or url_for('index'))

# ============================================================
# PART 4 — FINAL SOS ROUTES (KEEP ONLY THESE)
# ============================================================

from datetime import timezone   # add this import at top if missing


# ---------- SOS (send distress + GPS + notify contacts + notify police) ----------
@app.route('/sos', methods=['GET','POST'])
def sos():
    if 'user' not in session or session['user'].get('role') != 'user':
        flash('Login required','warning')
        return redirect(url_for('login_user'))

    # Fetch GPS (lat/lon) coming from home page auto-detect
    lat = request.values.get('lat') or ''
    lon = request.values.get('lon') or ''

    username = session['user']['name']
    user_name  = session['user'].get('name','Unknown')
    user_phone = session['user'].get('phone','')

    created = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    sos_id  = str(uuid.uuid4())[:8]
    status  = "active"

    # 1️⃣ SAVE ACTIVE SOS
    with open(ACTIVE_SOS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([sos_id, user_name, lat, lon, created, status])

    # 2️⃣ NOTIFY NEAREST POLICE
    nearest = assign_nearest_police(lat, lon)
    nid = str(uuid.uuid4())[:8]
    police_msg = f"SOS Alert from {user_name} ({user_phone}) at {lat},{lon}"

    with open(NOTIF_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([nid, '', nearest, police_msg, created, "1", "False"])  

    # 3️⃣ NOTIFY EMERGENCY CONTACTS
    # 3️⃣ NOTIFY EMERGENCY CONTACTS (FIXED)
    sms_text = f"EMERGENCY: {user_name} triggered SOS at {created}. Location: {lat},{lon}."
    user_id = session['user'].get('id')   # GET LOGGED-IN USER ID
    notify_emergency_contacts(user_id, sms_text)


    # 4️⃣ AJAX mode (when JS calls directly)
    if request.method == "POST" or request.is_json:
        return jsonify({"status":"ok", "sos_id":sos_id})

    # 5️⃣ REDIRECT MODE (normal browser)
    flash("SOS sent! Nearest police station alerted.", "danger")
    return redirect(url_for('user_dashboard'))


# ---------- SOS ACKNOWLEDGEMENT (Police marks SOS as resolved) ----------
@app.route('/sos/ack/<sosid>', methods=['POST'])
def sos_ack(sosid):
    rows = []
    try:
        with open(ACTIVE_SOS_CSV, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    except:
        rows = []

    changed = False
    for r in rows:
        if r.get('id') == sosid:
            r['status'] = "resolved"
            changed = True

    # Rewrite updated CSV
    with open(ACTIVE_SOS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id','name','lat','lon','created_at','status'])
        for r in rows:
            writer.writerow([
                r.get('id',''),
                r.get('name',''),
                r.get('lat',''),
                r.get('lon',''),
                r.get('created_at',''),
                r.get('status','')
            ])

    return jsonify({"status":"ok", "ack":changed})

# ============================================================
# API — RETURN ACTIVE SOS LIST (for live map)
# ============================================================
@app.route('/api/sos/active')
def api_sos_active():
    out = []
    try:
        with open(ACTIVE_SOS_CSV, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    except:
        rows = []

    for r in rows:
        if r.get('status') == 'active':
            out.append({
                "id": r.get("id"),
                "name": r.get("name"),
                "lat": float(r.get("lat") or 0),
                "lon": float(r.get("lon") or 0),
                "created_at": r.get("created_at")
            })

    return jsonify(out)

# ============================================================
# POLICE — LIVE SOS PAGE
# ============================================================
@app.route('/police/sos/live')
def police_live_sos():
    if 'user' not in session or session['user']['role'] != 'police':
        flash('Login required', 'warning')
        return redirect(url_for('login_police'))

    return render_template('police_sos_live.html', user=session['user'])

# ============================================================
# API — RESOLVE SOS
# ============================================================
@app.route('/api/sos/resolve/<sos_id>', methods=['POST'])
def api_sos_resolve(sos_id):
    rows = []
    try:
        with open(ACTIVE_SOS_CSV, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    except:
        return jsonify({"status": "error", "msg": "File missing"})

    changed = False
    for r in rows:
        if r.get('id') == sos_id:
            r['status'] = 'resolved'
            changed = True

    if changed:
        with open(ACTIVE_SOS_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id','name','lat','lon','created_at','status'])
            for r in rows:
                writer.writerow([
                    r.get('id',''),
                    r.get('name',''),
                    r.get('lat',''),
                    r.get('lon',''),
                    r.get('created_at',''),
                    r.get('status','')
                ])

    return jsonify({"status": "ok", "resolved": changed})

# ============================================================
# API — GET SOS HISTORY (all or user-specific)
# ============================================================
@app.route('/api/sos/history')
def api_sos_history():
    username = session['user']['name'] if 'user' in session else None
    role = session['user']['role'] if 'user' in session else None

    rows = []
    try:
        with open(ACTIVE_SOS_CSV, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    except:
        return jsonify([])

    # Police sees all SOS
    if role == "police":
        return jsonify(rows)

    # Users only see their own SOS
    if role == "user":
        filtered = [r for r in rows if r.get('name') == username]
        return jsonify(filtered)

    return jsonify([])

# ============================================================
# SOS HISTORY PAGES
# ============================================================

@app.route('/user/sos/history')
def user_sos_history():
    if 'user' not in session or session['user']['role'] != 'user':
        flash("Login required", "warning")
        return redirect(url_for('login_user'))
    return render_template("sos_history_user.html", user=session['user'])


@app.route('/police/sos/history')
def police_sos_history():
    if 'user' not in session or session['user']['role'] != 'police':
        flash("Login required", "warning")
        return redirect(url_for('login_police'))
    return render_template("sos_history_police.html", user=session['user'])

@app.route('/police/summary')
def police_summary():
    if 'user' not in session or session['user']['role'] != 'police':
        return redirect(url_for('login_police'))

    # Read reports
    with open(REPORTS_CSV, 'r', encoding='utf-8') as f:
        reports = list(csv.DictReader(f))


    # Logged-in police station
    station = session['user']['name']

    # Assigned reports
    assigned = [r for r in reports if r.get('assigned_police') == station]

    # Stats
    total_reports = len(reports)
    assigned_reports = len(assigned)
    in_progress = sum(1 for r in assigned if r.get('status') == 'In Progress')

    return render_template("police_summary.html",
                           total_reports=total_reports,
                           assigned_reports=assigned_reports,
                           in_progress=in_progress)


# ==========================
#  CRIME FORECAST API
# ==========================
@app.route('/api/forecast')
def api_forecast():
    days = int(request.args.get('days', 7))

    # --- Load reports ---
    try:
        df = pd.read_csv(REPORTS_CSV)
    except:
        return jsonify({"error": "No reports found"}), 400

    # Ensure date column exists
    if 'created_at' not in df.columns:
        return jsonify({"error": "created_at missing"}), 400

    # Convert to date only
    df['date'] = pd.to_datetime(df['created_at']).dt.date

    # Count crimes per day
    daily = df.groupby('date').size().reset_index(name='count')

    if len(daily) < 3:
        # Not enough data: return constant prediction
        forecast = []
        base = daily['count'].mean() if len(daily) else 1
        today = datetime.utcnow().date()
        for i in range(days):
            forecast.append({
                "date": str(today + timedelta(days=i)),
                "pred": round(base)
            })
        return jsonify(forecast)

    # Moving average forecast
    avg = daily['count'].rolling(window=3, min_periods=1).mean().iloc[-1]
    today = datetime.utcnow().date()

    forecast = []
    for i in range(days):
        forecast.append({
            "date": str(today + timedelta(days=i)),
            "pred": round(avg)
        })

    return jsonify(forecast)
@app.route('/police/community_reports')
def police_community_reports():
    if 'user' not in session or session['user'].get('role') != 'police':
        flash("Login required", "warning")
        return redirect(url_for('login_police'))

    reports = read_csv(REPORTS_CSV)

    return render_template("police_community_reports.html",
                           user=session['user'],
                           reports=reports)

# ---------------------------- # START APP # ---------------------------- 
if __name__ == '__main__': # ensure models loaded/trained (already executed above during import) 
    try: # attempt to retrain kmeans 
        train_kmeans() 
    except: 
        pass 
    # ===== start background scheduler to fetch news every 10 minutes =====
    try:
        scheduler = BackgroundScheduler()
    # run immediately once, then every 10 minutes
        scheduler.add_job(news_fetcher.fetch_and_store_news, 'interval', minutes=10, id='news_fetch_job', replace_existing=True)
    # run once at startup
        news_fetcher.fetch_and_store_news()
        scheduler.start()
        print("[scheduler] news job started")
    except Exception as e:
        print("Scheduler start failed:", e)
# ===== end scheduler =====

    app.run(host='0.0.0.0', port=5000, debug=True) 