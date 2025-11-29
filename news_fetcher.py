# news_fetcher.py
import os
import csv
import time
import re
import uuid
import requests
import feedparser
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

import spacy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
NEWS_CSV = os.path.join(DATA_FOLDER, "news.csv")
os.makedirs(DATA_FOLDER, exist_ok=True)

# ensure CSV
def ensure_csv(path, header):
    if not os.path.exists(path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f); writer.writerow(header)

ensure_csv(NEWS_CSV, ["id","title","description","location_text","lat","lon","source","url","published_at","is_hotspot","hotspot_group","created_at"])

# spaCy small model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    # If model not available the user must run: python -m spacy download en_core_web_sm
    raise RuntimeError("spaCy model not loaded. Run: python -m spacy download en_core_web_sm") from e

# geocoder (Nominatim + rate limiter)
geolocator = Nominatim(user_agent="crime_reporting_app_news")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0, max_retries=2, error_wait_seconds=2.0)

# Google News RSS query for crime in India (no API key)
RSS_URL = "https://news.google.com/rss/search?q=crime+India&hl=en-IN&gl=IN&ceid=IN:en"

def fetch_rss_articles(rss_url=RSS_URL, max_items=25):
    """Returns list of entries with title, summary, link, published"""
    feed = feedparser.parse(rss_url)
    out = []
    for e in feed.entries[:max_items]:
        title = e.get('title', '')
        summary = e.get('summary', '') or e.get('description','')
        link = e.get('link', '')
        published = e.get('published', '') or e.get('updated', '')
        out.append({"title": title, "description": summary, "url": link, "published": published, "source": e.get("source", {}).get("title") if isinstance(e.get("source"), dict) else None})
    return out

# extract location using NER (GPE/LOC/FAC) then simple regex fallback
def extract_location(text):
    if not text:
        return None
    doc = nlp(text)
    ents = [ent.text for ent in doc.ents if ent.label_ in ("GPE","LOC","FAC")]
    if ents:
        # prefer longest entity (heuristic)
        return max(ents, key=len).strip().strip('.,')
    # fallback: "in Koramangala" style
    m = re.search(r'\b(?:in|at|near|around|on)\s+([A-Z][a-zA-Z0-9\s,.-]{2,60})', text)
    if m:
        return m.group(1).strip().strip('.,')
    return None

# geocode to lat/lon (returns None,None if fails)
def geocode_place(place):
    if not place: 
        return None, None
    try:
        loc = geocode(place, exactly_one=True, timeout=10)
        if loc:
            return float(loc.latitude), float(loc.longitude)
    except Exception as e:
        print("Geocode error:", e)
    return None, None

# haversine distance
def haversine(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return 1e9
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

HOTSPOT_RADIUS_KM = 2.0
HOTSPOT_THRESHOLD = 3

def load_existing_news():
    rows = []
    try:
        with open(NEWS_CSV, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                rows.append(r)
    except:
        pass
    return rows

def write_news_row(row):
    with open(NEWS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([row.get(c,'') for c in ["id","title","description","location_text","lat","lon","source","url","published_at","is_hotspot","hotspot_group","created_at"]])

def mark_hotspot_for(new_lat, new_lon):
    rows = load_existing_news()
    nearby = []
    for r in rows:
        try:
            lat = float(r.get('lat') or 0); lon = float(r.get('lon') or 0)
            d = haversine(new_lat, new_lon, lat, lon)
            if d <= HOTSPOT_RADIUS_KM:
                nearby.append(r)
        except:
            continue
    if len(nearby) + 1 >= HOTSPOT_THRESHOLD:
        # assign group id
        existing_groups = [r.get('hotspot_group') for r in nearby if r.get('hotspot_group')]
        group = existing_groups[0] if existing_groups else str(uuid.uuid4())[:8]
        # update file: rewrite marking all nearby rows with group and is_hotspot
        all_rows = load_existing_news()
        updated = []
        for r in all_rows:
            if any(r.get('url') == n.get('url') for n in nearby) or (r.get('lat') and float(r.get('lat') or 0) and haversine(new_lat,new_lon,float(r.get('lat')),float(r.get('lon'))) <= HOTSPOT_RADIUS_KM):
                r['is_hotspot'] = "True"
                r['hotspot_group'] = group
            updated.append(r)
        # rewrite file
        with open(NEWS_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["id","title","description","location_text","lat","lon","source","url","published_at","is_hotspot","hotspot_group","created_at"])
            for r in updated:
                writer.writerow([r.get(c,'') for c in ["id","title","description","location_text","lat","lon","source","url","published_at","is_hotspot","hotspot_group","created_at"]])
        return group
    return None

def process_and_store_articles(articles):
    inserted = 0
    existing = load_existing_news()
    seen_keys = set((r.get('title','').strip() + '|' + (r.get('source') or '') ) for r in existing)
    for art in articles:
        title = art.get('title','').strip()
        desc = art.get('description','').strip()
        url = art.get('url','').strip()
        source = art.get('source') or ""
        published = art.get('published','')
        key = title + '|' + source
        if not title:
            continue
        if key in seen_keys:
            continue
        # Extract location and geocode
        loc_text = extract_location(title + ". " + desc)
        lat, lon = (None, None)
        if loc_text:
            lat, lon = geocode_place(loc_text)
        # If no lat/lon, try short fallback: look for major city names via geocode of title (could be noisy)
        if (lat is None or lon is None) and title:
            lat, lon = geocode_place(title.split(' - ')[-1])
        now = datetime.utcnow().isoformat()
        nid = str(uuid.uuid4())[:8]
        row = {
            "id": nid,
            "title": title,
            "description": desc,
            "location_text": loc_text or "",
            "lat": lat or "",
            "lon": lon or "",
            "source": source or "",
            "url": url,
            "published_at": published or "",
            "is_hotspot": "False",
            "hotspot_group": "",
            "created_at": now
        }
        write_news_row(row)
        if lat and lon:
            mark_hotspot_for(float(lat), float(lon))
        inserted += 1
        seen_keys.add(key)
    return inserted

def fetch_and_store_news():
    try:
        articles = fetch_rss_articles()
        cnt = process_and_store_articles(articles)
        print(f"[news_fetcher] inserted {cnt} news at {datetime.utcnow().isoformat()}")
        return cnt
    except Exception as e:
        print("Error in fetch_and_store_news:", e)
        return 0

# quick test main
if __name__ == "__main__":
    print("Fetching news...")
    print(fetch_and_store_news())
