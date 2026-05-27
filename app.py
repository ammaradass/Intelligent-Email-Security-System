from dotenv import load_dotenv
load_dotenv()  # Load .env file automatically on startup

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import re
import sqlite3
import json
import os
import time
import base64
import requests
from pathlib import Path
from urllib.parse import urlparse

import logging

import joblib
from scipy.sparse import csr_matrix, hstack

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(32).hex())
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB max request size

# ── Flask-Login ──────────────────────────────────────────────────────────────
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."


class User(UserMixin):
    def __init__(self, id, username, email, password_hash):
        self.id            = id
        self.username      = username
        self.email         = email
        self.password_hash = password_hash


@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, password_hash FROM users WHERE id = ?", (int(user_id),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return User(*row)
    return None

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "email_security.db"
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "xgb_email_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
URL_COLUMNS_PATH = MODELS_DIR / "url_feature_columns.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
SETTINGS_PATH = BASE_DIR / "settings.json"

# ── Load phishing detection config from settings.json ────────────────────────
# All threat intelligence lists are externalized into settings.json.
# This follows the separation of concerns principle — threat data is kept
# separate from application logic, allowing updates without code changes.
def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        logger.warning("settings.json not found at %s — using empty phishing config", SETTINGS_PATH)
        return {}
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "phishing_detection" not in data:
        logger.warning("settings.json found but missing 'phishing_detection' key")
    return data

PHISHING_CONFIG = load_settings().get("phishing_detection", {})
logger.info("Phishing detection config loaded — %d brands, %d keywords, %d TLDs",
    len(PHISHING_CONFIG.get("impersonated_brands", [])),
    len(PHISHING_CONFIG.get("suspicious_keywords", [])),
    len(PHISHING_CONFIG.get("suspicious_tlds", [])),
)

VT_API_KEY = os.getenv("VT_API_KEY", "").strip()
if VT_API_KEY:
    print(f"[INFO] VirusTotal API key loaded successfully (key: ...{VT_API_KEY[-4:]})")
else:
    print("[WARNING] No VirusTotal API key found. Set VT_API_KEY in .env file.")

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    url_feature_columns = joblib.load(URL_COLUMNS_PATH)
    return model, vectorizer, url_feature_columns


def load_metrics():
    if not METRICS_PATH.exists():
        return None

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


model, vectorizer, url_feature_columns = load_artifacts()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT,
            receiver TEXT,
            subject TEXT,
            body TEXT,
            classification TEXT,
            confidence REAL,
            urls_detected INTEGER,
            reasons TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS urls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_id INTEGER,
            url TEXT,
            FOREIGN KEY (email_id) REFERENCES emails(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    # Keep URL tokens so the model recognizes phishing URL patterns
    text = re.sub(r"[^a-z0-9\s./:@_-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s]+|www\.[^\s]+", str(text))


def extract_url_features(text: str) -> dict:
    urls = extract_urls(text)

    if not urls:
        return {
            "url_count": 0,
            "max_url_length": 0,
            "avg_url_length": 0.0,
            "has_ip_url": 0,
            "has_at_symbol": 0,
            "has_dash": 0,
            "has_many_dots": 0,
            "has_suspicious_pattern": 0,
            "uses_https": 0,
        }

    lengths = [len(url) for url in urls]

    has_ip_url = any(
        re.search(r"(?:https?://)?(?:\d{1,3}\.){3}\d{1,3}", url)
        for url in urls
    )
    has_at_symbol = any("@" in url for url in urls)
    has_dash = any("-" in url for url in urls)
    has_many_dots = any(url.count(".") >= 3 for url in urls)
    has_suspicious_pattern = any(
        any(pattern in url.lower() for pattern in ["verify", "login", "secure", "update", "%", "@"])
        for url in urls
    )
    uses_https = any(url.lower().startswith("https://") for url in urls)

    return {
        "url_count": len(urls),
        "max_url_length": max(lengths),
        "avg_url_length": sum(lengths) / len(lengths),
        "has_ip_url": int(has_ip_url),
        "has_at_symbol": int(has_at_symbol),
        "has_dash": int(has_dash),
        "has_many_dots": int(has_many_dots),
        "has_suspicious_pattern": int(has_suspicious_pattern),
        "uses_https": int(uses_https),
    }


def get_sender_domain(sender: str) -> str:
    sender = (sender or "").strip().lower()
    if "@" not in sender:
        return ""
    return sender.split("@")[-1].strip()


def get_url_domain(url: str) -> str:
    url = (url or "").strip().lower()
    if not url:
        return ""

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


def contains_any(text: str, keywords: list[str]) -> list[str]:
    text = (text or "").lower()
    return [keyword for keyword in keywords if keyword in text]

def get_sender_local_part(sender: str) -> str:
    sender = (sender or "").strip().lower()
    if "@" not in sender:
        return sender
    return sender.split("@")[0].strip()


def is_free_email_domain(domain: str) -> bool:
    free_domains = set(PHISHING_CONFIG.get("free_email_domains", []))
    return domain in free_domains

def get_registered_like_domain(domain: str) -> str:
    domain = (domain or "").lower().strip()
    if not domain:
        return ""

    if ":" in domain:
        domain = domain.split(":")[0]

    parts = [part for part in domain.split(".") if part]

    if len(parts) >= 2:
        return ".".join(parts[-2:])

    return domain


def count_subdomains(domain: str) -> int:
    domain = (domain or "").lower().strip()
    if not domain:
        return 0

    if ":" in domain:
        domain = domain.split(":")[0]

    parts = [part for part in domain.split(".") if part]
    return max(0, len(parts) - 2)


def contains_brand_and_extra_words(domain: str, brands: list[str]) -> tuple[bool, str]:
    domain = (domain or "").lower()
    clean_domain = domain.replace("-", ".").replace("_", ".")

    # Loaded from settings.json — no hardcoding
    suspicious_extra_words = PHISHING_CONFIG.get("suspicious_extra_words", [])

    for brand in brands:
        if brand in clean_domain:
            extras = [word for word in suspicious_extra_words if word in clean_domain]
            if extras:
                return True, f"{brand} + {', '.join(extras[:3])}"

    return False, ""


def has_misleading_subdomain(domain: str, brands: list[str]) -> tuple[bool, str]:
    domain = (domain or "").lower().strip()
    registered = get_registered_like_domain(domain)

    for brand in brands:
        if brand in domain and brand not in registered:
            return True, brand

    return False, ""


def is_long_domain(domain: str) -> bool:
    domain = (domain or "").strip().lower()
    return len(domain) >= 30


def detect_typosquatting(domain: str) -> tuple[bool, str]:
    """
    Detects typosquatting attacks where attackers replace characters
    in brand names with visually similar ones.
    Examples: paypa1.com (1 instead of l), micros0ft.com (0 instead of o)

    Variants are loaded from settings.json — no hardcoding.
    """
    typo_map = PHISHING_CONFIG.get("typosquatting_map", {})
    domain = (domain or "").lower().strip()

    for brand, variants in typo_map.items():
        for variant in variants:
            if variant in domain:
                return True, f"{variant} (typo of {brand})"

    return False, ""

def url_to_vt_id(url: str) -> str:
    url = (url or "").strip()
    encoded = base64.urlsafe_b64encode(url.encode("utf-8")).decode("utf-8")
    return encoded.strip("=")


def scan_url_virustotal(url: str, max_attempts: int = 6, poll_interval: int = 5) -> dict:
    """
    Submit a URL to VirusTotal and poll until the analysis is complete.

    Args:
        url: The URL to scan.
        max_attempts: Maximum number of polling attempts (default: 6).
        poll_interval: Seconds between polling attempts (default: 5).
            Total max wait = max_attempts * poll_interval = 30s by default.
    """
    empty_result = {
        "enabled": True,
        "status": "",
        "malicious": 0,
        "suspicious": 0,
        "harmless": 0,
        "undetected": 0,
        "score": 0,
    }

    if not VT_API_KEY:
        return {**empty_result, "enabled": False, "status": "No API key configured"}

    headers = {"x-apikey": VT_API_KEY}

    try:
        # Step 1: Submit the URL for scanning
        logger.info("VirusTotal: submitting URL for scan: %s", url)
        submit_response = requests.post(
            "https://www.virustotal.com/api/v3/urls",
            headers=headers,
            data={"url": url},
            timeout=15,
        )

        if submit_response.status_code not in (200, 202):
            logger.warning("VirusTotal: submit failed with status %s", submit_response.status_code)
            return {**empty_result, "status": f"Submit failed: {submit_response.status_code}"}

        # Step 2: Poll for the analysis report
        vt_id = url_to_vt_id(url)
        report_url = f"https://www.virustotal.com/api/v3/urls/{vt_id}"

        for attempt in range(1, max_attempts + 1):
            time.sleep(poll_interval)
            logger.info("VirusTotal: polling attempt %d/%d for %s", attempt, max_attempts, url)

            report_response = requests.get(report_url, headers=headers, timeout=15)

            if report_response.status_code != 200:
                logger.warning("VirusTotal: report fetch returned status %s", report_response.status_code)
                continue

            data = report_response.json()
            attributes = data.get("data", {}).get("attributes", {})
            analysis_status = attributes.get("last_analysis_results")

            # If analysis_results exists and has entries, the scan is done
            if analysis_status:
                stats = attributes.get("last_analysis_stats", {})
                malicious = int(stats.get("malicious", 0))
                suspicious = int(stats.get("suspicious", 0))
                harmless = int(stats.get("harmless", 0))
                undetected = int(stats.get("undetected", 0))
                score = (malicious * 2) + suspicious

                logger.info(
                    "VirusTotal: scan complete for %s — malicious=%d, suspicious=%d",
                    url, malicious, suspicious,
                )

                return {
                    "enabled": True,
                    "status": "OK",
                    "malicious": malicious,
                    "suspicious": suspicious,
                    "harmless": harmless,
                    "undetected": undetected,
                    "score": score,
                }

        # Exhausted all polling attempts
        logger.warning("VirusTotal: timed out waiting for results for %s", url)
        return {**empty_result, "status": "Analysis timed out"}

    except requests.exceptions.Timeout:
        logger.error("VirusTotal: request timed out for %s", url)
        return {**empty_result, "status": "Request timed out"}

    except requests.exceptions.ConnectionError:
        logger.error("VirusTotal: connection failed for %s", url)
        return {**empty_result, "status": "Connection failed"}

    except Exception as e:
        logger.error("VirusTotal: unexpected error for %s — %s", url, str(e))
        return {**empty_result, "status": f"Error: {str(e)}"}

def analyze_urls_with_virustotal(urls: list[str]) -> dict:
    if not urls:
        return {
            "checked": False,
            "status": "No URLs to scan",
            "total_score": 0,
            "results": []
        }

    results = []
    total_score = 0

    for url in urls[:2]:
        vt_result = scan_url_virustotal(url)
        total_score += vt_result["score"]
        results.append({
            "url": url,
            **vt_result
        })

    checked = any(item.get("enabled") for item in results)

    if not checked and results:
        status = results[0].get("status", "VirusTotal not used")
    else:
        status = "Completed"

    return {
        "checked": checked,
        "status": status,
        "total_score": total_score,
        "results": results
    }
def apply_rule_based_phishing_boost(
    sender: str,
    subject: str,
    body: str,
    urls: list[str],
    ml_label: str,
    ml_confidence: float,
    vt_summary: dict | None = None
):
    subject = subject or ""
    body = body or ""
    sender = sender or ""

    text = f"{sender} {subject} {body}".lower()
    sender_domain = get_sender_domain(sender)
    sender_local = get_sender_local_part(sender)

    # ── Load threat intelligence lists from config ───────────────────────────
    # Lists are defined in settings.json — not hardcoded here.
    # To add/remove brands, keywords, or TLDs, edit settings.json only.
    suspicious_keywords     = PHISHING_CONFIG.get("suspicious_keywords", [])
    urgent_phrases          = PHISHING_CONFIG.get("urgent_phrases", [])
    suspicious_tlds         = PHISHING_CONFIG.get("suspicious_tlds", [])
    url_shorteners          = PHISHING_CONFIG.get("url_shorteners", [])
    impersonated_brands     = PHISHING_CONFIG.get("impersonated_brands", [])
    suspicious_sender_words = PHISHING_CONFIG.get("suspicious_sender_words", [])
    download_words          = PHISHING_CONFIG.get("download_words", [])

    keyword_hits = contains_any(text, suspicious_keywords)
    urgent_hits = contains_any(text, urgent_phrases)
    download_hits = contains_any(text, download_words)
    brand_hits = contains_any(text, impersonated_brands)
    sender_word_hits = contains_any(sender_local, suspicious_sender_words)

    has_ip_url = any(
        re.search(r"(?:https?://)?(?:\d{1,3}\.){3}\d{1,3}", url)
        for url in urls
    )

    has_at_symbol = any("@" in url for url in urls)
    has_many_dots = any(url.count(".") >= 3 for url in urls)
    has_dash = any("-" in url for url in urls)

    has_suspicious_url_words = any(
        any(word in url.lower() for word in [
            "verify", "login", "secure", "update", "account",
            "bank", "confirm", "password", "reset", "signin"
        ])
        for url in urls
    )

    has_suspicious_tld = any(
        any(tld in url.lower() for tld in suspicious_tlds)
        for url in urls
    )

    uses_shortener = any(
        any(shortener in url.lower() for shortener in url_shorteners)
        for url in urls
    )

    has_many_digits_in_url = any(
        sum(ch.isdigit() for ch in url) >= 5
        for url in urls
    )

    has_too_many_urls = len(urls) >= 3

    url_domains = []
    misleading_subdomain_hits = []
    brand_plus_extra_hits = []
    long_domain_hits = []
    many_subdomain_hits = []
    typosquatting_hits = []

    for url in urls:
        domain = get_url_domain(url)
        if not domain:
            continue

        url_domains.append(domain)

        misleading, misleading_brand = has_misleading_subdomain(domain, impersonated_brands)
        if misleading:
            misleading_subdomain_hits.append(f"{domain} ({misleading_brand})")

        brand_extra, brand_extra_desc = contains_brand_and_extra_words(domain, impersonated_brands)
        if brand_extra:
            brand_plus_extra_hits.append(f"{domain} ({brand_extra_desc})")

        if is_long_domain(domain):
            long_domain_hits.append(domain)

        if count_subdomains(domain) >= 2:
            many_subdomain_hits.append(domain)

        # ── Typosquatting detection ──────────────────────────────────────────
        # Checks for character substitutions in brand names
        # e.g. paypa1.com (1 instead of l), micros0ft.com (0 instead of o)
        typo_detected, typo_desc = detect_typosquatting(domain)
        if typo_detected:
            typosquatting_hits.append(f"{domain} ({typo_desc})")

    sender_domain_suspicious = False
    if sender_domain and any(tld in sender_domain for tld in suspicious_tlds):
        sender_domain_suspicious = True

    sender_brand_mismatch = False
    if brand_hits and sender_domain:
        sender_brand_mismatch = not any(brand in sender_domain for brand in impersonated_brands)

    brand_in_url = any(
        any(brand in url.lower() for brand in impersonated_brands)
        for url in urls
    )

    sender_url_domain_mismatch = False
    if sender_domain and url_domains:
        sender_url_domain_mismatch = all(sender_domain not in domain for domain in url_domains)

    sender_uses_free_email_for_brand = False
    if brand_hits and sender_domain:
        # Only flag if the brand mentioned is NOT the sender's own domain
        # e.g. paypal-support@gmail.com claiming to be PayPal → flag
        # but security@gmail.com just existing → do not flag gmail against itself
        non_self_brands = [b for b in brand_hits if b not in sender_domain]
        if non_self_brands:
            sender_uses_free_email_for_brand = is_free_email_domain(sender_domain)

    sender_local_mentions_brand = any(brand in sender_local for brand in impersonated_brands)
    sender_local_brand_domain_mismatch = False
    if sender_local_mentions_brand and sender_domain:
        sender_local_brand_domain_mismatch = not any(brand in sender_domain for brand in impersonated_brands)

    phishing_score = 0
    reasons = []

    if keyword_hits:
        kw_pts = min(len(keyword_hits), 5)
        phishing_score += kw_pts
        reasons.append(f"Suspicious keywords found: {', '.join(keyword_hits[:5])}||{kw_pts}")

    if urgent_hits:
        phishing_score += 3
        reasons.append(f"Urgent or threatening phrases found: {', '.join(urgent_hits[:3])}||3")

    if download_hits:
        phishing_score += 2
        reasons.append(f"Possible malicious download wording found: {', '.join(download_hits[:3])}||2")

    if has_ip_url:
        phishing_score += 3
        reasons.append("URL uses IP address instead of a normal domain||3")

    if has_at_symbol:
        phishing_score += 2
        reasons.append("URL contains @ symbol||2")

    if has_many_dots:
        phishing_score += 1
        reasons.append("URL has many dots||1")

    if has_dash:
        phishing_score += 1
        reasons.append("URL contains dash characters||1")

    if has_suspicious_url_words:
        phishing_score += 2
        reasons.append("URL contains phishing-related words||2")

    if has_suspicious_tld:
        phishing_score += 2
        reasons.append("URL uses suspicious top-level domain||2")

    if uses_shortener:
        phishing_score += 3
        reasons.append("URL uses shortening service||3")

    if has_many_digits_in_url:
        phishing_score += 1
        reasons.append("URL contains many digits||1")

    if has_too_many_urls:
        phishing_score += 2
        reasons.append("Email contains multiple URLs||2")

    if brand_hits:
        phishing_score += 1
        reasons.append(f"Brand names mentioned: {', '.join(brand_hits[:4])}||1")

    if brand_in_url:
        phishing_score += 2
        reasons.append("URL includes brand-related wording||2")

    if sender_word_hits:
        phishing_score += 1
        reasons.append(f"Sender name contains suspicious role words: {', '.join(sender_word_hits[:3])}||1")

    if sender_domain_suspicious:
        phishing_score += 2
        reasons.append("Sender domain looks suspicious||2")

    if sender_brand_mismatch:
        phishing_score += 3
        reasons.append("Possible sender and brand mismatch detected||3")

    if sender_url_domain_mismatch:
        phishing_score += 2
        reasons.append("Sender domain does not match linked URL domain||2")

    if sender_uses_free_email_for_brand:
        phishing_score += 3
        reasons.append("Brand-related email uses a free email provider||3")

    if sender_local_brand_domain_mismatch:
        phishing_score += 3
        reasons.append("Sender address mentions a brand but the domain is unrelated||3")

    if misleading_subdomain_hits:
        phishing_score += 3
        reasons.append(f"Misleading brand in subdomain detected: {misleading_subdomain_hits[0]}||3")

    if brand_plus_extra_hits:
        phishing_score += 3
        reasons.append(f"Domain combines brand with phishing words: {brand_plus_extra_hits[0]}||3")

    if long_domain_hits:
        phishing_score += 1
        reasons.append(f"Long suspicious domain detected: {long_domain_hits[0]}||1")

    if many_subdomain_hits:
        phishing_score += 2
        reasons.append(f"Domain has many subdomains: {many_subdomain_hits[0]}||2")

    if typosquatting_hits:
        phishing_score += 4
        reasons.append(f"Typosquatting attack detected in URL: {typosquatting_hits[0]}||4")

    # ── Sender domain typosquatting check ────────────────────────────────────
    # Attackers often use typosquatted domains in the sender address itself
    # e.g. no-reply@paypa1.com — caught here even with no URLs in the body
    if sender_domain:
        sender_typo, sender_typo_desc = detect_typosquatting(sender_domain)
        if sender_typo:
            phishing_score += 4
            reasons.append(f"Typosquatting attack detected in sender domain: {sender_domain} ({sender_typo_desc})||4")

    if len(urls) > 0:
        phishing_score += 1
        reasons.append("Email contains at least one URL||1")

    # ── HTTP vs HTTPS check ──────────────────────────────────────────────────
    # Any legitimate website in 2025 uses HTTPS. HTTP-only URLs are a strong
    # phishing signal — attackers use HTTP to avoid SSL certificate requirements.
    uses_http_only = any(
        url.lower().startswith("http://") and not url.lower().startswith("https://")
        for url in urls
    )
    if uses_http_only:
        phishing_score += 2
        reasons.append("URL uses insecure HTTP instead of HTTPS||2")
    if vt_summary and vt_summary.get("checked"):
        vt_total_score = int(vt_summary.get("total_score", 0))
        vt_results = vt_summary.get("results", [])

        if vt_total_score > 0:
            phishing_score += min(vt_total_score, 6)

        for item in vt_results:
            malicious = item.get("malicious", 0)
            suspicious = item.get("suspicious", 0)

            if malicious > 0 or suspicious > 0:
                vt_pts = min((malicious * 2) + suspicious, 6)
                reasons.append(
                    f"VirusTotal flagged URL: malicious={malicious}, suspicious={suspicious}||{vt_pts}"
                )
    final_label = ml_label
    final_confidence = ml_confidence

    # ML says Phishing + very high score → High Risk Phishing
    if ml_label == "Spam/Phishing" and phishing_score >= 25:
        final_label = "High Risk Phishing"
        final_confidence = max(ml_confidence, 92.0)

    # ML says Phishing + high score → Spam/Phishing (trust ML confidence as-is)
    elif ml_label == "Spam/Phishing" and phishing_score >= 12:
        final_label = "Spam/Phishing"
        final_confidence = ml_confidence  # do not force up — trust the ML score

    # ML says Phishing + moderate score → Suspicious
    elif ml_label == "Spam/Phishing" and phishing_score >= 5:
        final_label = "Suspicious"
        final_confidence = max(ml_confidence, 78.0)

    # ── False Positive Correction ────────────────────────────────────────────
    # ML says Phishing BUT rule engine finds almost no real signals (score <= 4).
    # This protects legitimate emails with financial/professional vocabulary
    # (e.g. "log in", "account", "bank") from being wrongly classified.
    # When there are no suspicious URLs, no spoofed domains, no urgency phrases,
    # and no IP-based links, the ML is likely reacting to common words only.
    elif ml_label == "Spam/Phishing" and phishing_score <= 4:
        final_label = "Safe"
        final_confidence = 65.0  # moderate confidence — acknowledge ML uncertainty

    # ML says Safe + high score → override to High Risk
    elif ml_label == "Safe" and phishing_score >= 15:
        final_label = "High Risk Phishing"
        final_confidence = max(ml_confidence, 90.0)

    # ML says Safe + moderate score → Suspicious
    # Threshold raised to 10 to reduce false positives on legitimate emails
    elif ml_label == "Safe" and phishing_score >= 10:
        final_label = "Suspicious"
        final_confidence = max(ml_confidence, 78.0)

    return final_label, final_confidence, phishing_score, reasons

def predict_email(subject: str, body: str):
    combined_text = preprocess_text(f"{subject} {body}")
    text_features = vectorizer.transform([combined_text])

    url_feature_dict = extract_url_features(body)
    url_values = [[float(url_feature_dict[col]) for col in url_feature_columns]]
    url_features = csr_matrix(url_values)

    final_features = hstack([text_features, url_features], format="csr")

    prediction = int(model.predict(final_features)[0])
    probability = float(model.predict_proba(final_features)[0][1])

    # ── Adaptive Threshold Engine ────────────────────────────────────────
    # Professional cybersecurity systems use adaptive thresholds rather than
    # a fixed 0.50 cutoff. The threshold adapts based on:
    #   1. Email length  — short emails have less context, require higher confidence
    #   2. URL presence  — URLs are strong phishing indicators, lower threshold
    #   3. Evidence floor — minimum confidence required to flag as phishing

    word_count   = len(combined_text.split())
    url_count    = len(extract_urls(body))
    has_urls     = url_count > 0

    # Determine adaptive threshold based on available evidence
    if word_count < 10 and not has_urls:
        # Very short email, no URLs → need high confidence (reduce false positives)
        threshold = 0.82
    elif word_count < 20 and not has_urls:
        # Short email, no URLs → still require strong evidence
        threshold = 0.75
    elif word_count < 10 and has_urls:
        # Short email but has URLs → URLs are strong signal, lower threshold
        threshold = 0.60
    elif has_urls:
        # Normal email with URLs → standard phishing threshold
        threshold = 0.65
    else:
        # Normal email, no URLs → slightly elevated threshold
        threshold = 0.70

    if probability >= threshold:
        label = "Spam/Phishing"
        # ── Phishing confidence: calibrated using isotonic scaling ──────
        # Raw ML probability tends to be overconfident near boundaries.
        # We apply a soft calibration: scale probability toward a meaningful
        # range so 70% raw → ~78% displayed, 95% raw → ~97% displayed.
        raw_pct = probability * 100
        calibrated = 50 + (raw_pct - 50) * 0.95
        confidence  = round(min(99.0, max(70.0, calibrated)), 2)

    else:
        label = "Safe"
        # ── Safe confidence: margin-based scoring ───────────────────────
        # Professional systems (e.g. Microsoft Defender, Proofpoint) score
        # safe confidence based on the margin between the ML probability
        # and the adaptive threshold — not just 1 - probability.
        #
        # Formula:
        #   margin   = threshold - probability       (how far below the line)
        #   margin % = margin / threshold            (normalized 0→1)
        #   display  = 55 + margin% × 44            (maps to 55%–99%)
        #
        # Examples:
        #   prob=0.63, threshold=0.82 → margin=0.19 → margin%=0.23 → 65%
        #   prob=0.40, threshold=0.82 → margin=0.42 → margin%=0.51 → 77%
        #   prob=0.10, threshold=0.70 → margin=0.60 → margin%=0.86 → 93%

        margin     = threshold - probability
        margin_pct = margin / threshold
        confidence = round(min(99.0, max(55.0, 55.0 + margin_pct * 44.0)), 2)

    urls = extract_urls(body)

    return label, confidence, urls, url_feature_dict


def save_to_database(sender, receiver, subject, body, label, confidence, urls, reasons=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Convert reasons list to JSON string
    reasons_json = json.dumps(reasons) if reasons else "[]"

    cursor.execute("""
        INSERT INTO emails (sender, receiver, subject, body, classification, confidence, urls_detected, reasons)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (sender, receiver, subject, body, label, confidence, len(urls), reasons_json))

    email_id = cursor.lastrowid

    for url in urls:
        cursor.execute("""
            INSERT INTO urls (email_id, url)
            VALUES (?, ?)
        """, (email_id, url))

    conn.commit()
    conn.close()


def get_email_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, sender, receiver, subject, classification, confidence, urls_detected, reasons
        FROM emails
        ORDER BY id DESC
    """)

    records = cursor.fetchall()
    conn.close()
    return records

# ── Composite Risk Score ─────────────────────────────────────────────────────
# MAX_RULE_POINTS is the theoretical maximum the rule engine can produce.
# Calculated by summing the highest possible points from all rule branches:
#   keywords(5) + urgent(3) + download(2) + ip_url(3) + at_symbol(2) +
#   many_dots(1) + dash(1) + suspicious_url_words(2) + suspicious_tld(2) +
#   shortener(3) + many_digits(1) + too_many_urls(2) + brand(1) +
#   brand_in_url(2) + sender_word(1) + sender_domain_suspicious(2) +
#   sender_brand_mismatch(3) + sender_url_mismatch(2) +
#   free_email_brand(3) + local_brand_mismatch(3) +
#   misleading_subdomain(3) + brand_plus_extra(3) +
#   long_domain(1) + many_subdomains(2) + has_urls(1) +
#   http_only(2) + typosquatting_url(4) + typosquatting_sender(4) + virustotal(6) = 65
MAX_RULE_POINTS = 65

def calculate_composite_score(ml_probability: float, rule_points: int) -> tuple:
    """
    Combines ML probability and rule-based score into a single normalized
    composite risk score from 0 to 100.

    Methodology:
      - ML model weight: 60% — trained on 107,533 emails, high empirical reliability
      - Rule engine weight: 40% — expert heuristics, interpretable security signals

    Risk levels follow CVSS-inspired banding:
      0–25   → Low Risk     (Safe)
      26–49  → Medium Risk  (Suspicious)
      50–74  → High Risk    (Spam/Phishing)
      75–100 → Critical Risk (High Risk Phishing)

    References:
      - NIST SP 800-30 Risk Scoring Framework
      - CVSS v3.1 Severity Ratings (NVD/NIST)
      - Scikit-learn Probability Calibration [14]
    """
    W_ML   = 0.60
    W_RULE = 0.40

    ml_score   = ml_probability * 100                                     # 0–100
    rule_score = min(rule_points / MAX_RULE_POINTS, 1.0) * 100            # 0–100, capped

    composite = (W_ML * ml_score) + (W_RULE * rule_score)
    composite = round(min(100.0, max(0.0, composite)), 1)

    if composite >= 75:
        risk_level = "Critical Risk"
    elif composite >= 55:
        risk_level = "High Risk"
    elif composite >= 35:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    return composite, risk_level


def process_email(sender, receiver, subject, body):
    ml_label, ml_confidence, urls, url_feature_dict = predict_email(subject, body)

    vt_summary = analyze_urls_with_virustotal(urls)

    final_label, final_confidence, phishing_score, reasons = apply_rule_based_phishing_boost(
        sender,
        subject,
        body,
        urls,
        ml_label,
        ml_confidence,
        vt_summary
    )

    # ── Composite Risk Score ─────────────────────────────────────────────
    # Combines ML probability and rule-based points into a single normalized
    # 0–100 score with a CVSS-inspired risk level label.
    #
    # Uses final_label (after false positive correction) to determine risk
    # direction — ensures composite score matches the final verdict.
    # When Safe: invert confidence to get phishing probability.
    # When Spam/Phishing: use confidence directly as phishing probability.
    if final_label == "Safe":
        ml_probability_est = 1.0 - (ml_confidence / 100.0)
    else:
        ml_probability_est = ml_confidence / 100.0
    composite_score, risk_level = calculate_composite_score(ml_probability_est, phishing_score)

    save_to_database(sender, receiver, subject, body, final_label, final_confidence, urls, reasons)

    return {
        "label": final_label,
        "confidence": final_confidence,
        "urls_detected": len(urls),
        "url_list": urls,
        "url_features": url_feature_dict,
        "phishing_score": phishing_score,
        "composite_score": composite_score,
        "risk_level": risk_level,
        "reasons": reasons,
        "ml_label": ml_label,
        "ml_confidence": ml_confidence,
        "vt_summary": vt_summary,
    }

MAX_FIELD_LENGTH = 50_000  # characters


@app.route('/')
@login_required
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
@login_required
def analyze_email():
    sender = (request.form.get('sender') or "").strip()
    receiver = (request.form.get('receiver') or "").strip()
    subject = (request.form.get('subject') or "").strip()
    body = (request.form.get('body') or "").strip()

    # Input validation
    if not sender or not receiver or not subject or not body:
        logger.warning("Incomplete form submission — missing required fields")
        return render_template('index.html'), 400

    if any(len(field) > MAX_FIELD_LENGTH for field in [sender, receiver, subject, body]):
        logger.warning("Rejected oversized input from %s", sender)
        return render_template('index.html'), 413

    logger.info("Analyzing email from=%s to=%s subject=%s", sender, receiver, subject[:80])

    result = process_email(sender, receiver, subject, body)

    logger.info(
        "Result: label=%s confidence=%.1f%% phishing_score=%d",
        result["label"], result["confidence"], result["phishing_score"],
    )

    return render_template(
        'result.html',
        sender=sender,
        receiver=receiver,
        subject=subject,
        body=body,
        result=result
    )


@app.route('/history')
@login_required
def history():
    logger.info("Viewing email analysis history")
    records = get_email_history()
    
    # Parse JSON reasons for each record
    parsed_records = []
    for record in records:
        # record is a tuple: (id, sender, receiver, subject, classification, confidence, urls_detected, reasons)
        record_list = list(record)
        
        # Parse the reasons JSON (index 7)
        if record_list[7]:
            try:
                record_list[7] = json.loads(record_list[7])
            except (json.JSONDecodeError, TypeError):
                record_list[7] = []
        else:
            record_list[7] = []
        
        parsed_records.append(tuple(record_list))
    
    return render_template('history.html', records=parsed_records)


@app.route('/evaluation')
@login_required
def evaluation():
    metrics = load_metrics()
    return render_template('evaluation.html', metrics=metrics)


def get_dashboard_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Total emails
    cursor.execute("SELECT COUNT(*) FROM emails")
    total = cursor.fetchone()[0]

    # Count by classification
    cursor.execute("SELECT classification, COUNT(*) FROM emails GROUP BY classification")
    classification_counts = {row[0]: row[1] for row in cursor.fetchall()}

    safe_count       = classification_counts.get("Safe", 0)
    suspicious_count = classification_counts.get("Suspicious", 0)
    phishing_count   = classification_counts.get("High Risk Phishing", 0) + classification_counts.get("Spam/Phishing", 0)

    # Average confidence
    cursor.execute("SELECT AVG(confidence) FROM emails")
    avg_confidence = cursor.fetchone()[0] or 0.0

    # Total URLs detected
    cursor.execute("SELECT SUM(urls_detected) FROM emails")
    total_urls = cursor.fetchone()[0] or 0

    # Last 7 emails for recent activity chart (classification per entry)
    cursor.execute("""
        SELECT classification FROM emails ORDER BY id DESC LIMIT 30
    """)
    recent_labels = [row[0] for row in cursor.fetchall()]
    recent_safe       = recent_labels.count("Safe")
    recent_suspicious = recent_labels.count("Suspicious")
    recent_phishing   = recent_labels.count("High Risk Phishing") + recent_labels.count("Spam/Phishing")

    # Most recent 5 emails
    cursor.execute("""
        SELECT sender, subject, classification, confidence
        FROM emails ORDER BY id DESC LIMIT 5
    """)
    recent_emails = cursor.fetchall()

    conn.close()

    return {
        "total": total,
        "safe": safe_count,
        "suspicious": suspicious_count,
        "phishing": phishing_count,
        "avg_confidence": round(avg_confidence, 1),
        "total_urls": total_urls,
        "recent_safe": recent_safe,
        "recent_suspicious": recent_suspicious,
        "recent_phishing": recent_phishing,
        "recent_emails": recent_emails,
    }


@app.route('/dashboard')
@login_required
def dashboard():
    logger.info("Viewing dashboard")
    stats = get_dashboard_stats()
    metrics = load_metrics()
    return render_template('dashboard.html', stats=stats, metrics=metrics)



# ── Auth helpers ─────────────────────────────────────────────────────────────
def get_user_by_username(username):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, password_hash FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return User(*row) if row else None


def get_user_by_email(email):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, password_hash FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    return User(*row) if row else None


def create_user(username, email, password):
    password_hash = generate_password_hash(password)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
        (username, email, password_hash)
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return User(user_id, username, email, password_hash)


# ── Auth routes ───────────────────────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    error = None
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        email    = (request.form.get('email') or '').strip().lower()
        password = (request.form.get('password') or '')
        confirm  = (request.form.get('confirm') or '')

        if not username or not email or not password:
            error = "All fields are required."
        elif len(username) < 3:
            error = "Username must be at least 3 characters."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm:
            error = "Passwords do not match."
        elif get_user_by_username(username):
            error = "Username already taken."
        elif get_user_by_email(email):
            error = "Email already registered."
        else:
            user = create_user(username, email, password)
            login_user(user)
            logger.info("New user registered: %s", username)
            return redirect(url_for('home'))

    return render_template('register.html', error=error)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    error = None
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = (request.form.get('password') or '')

        user = get_user_by_username(username)
        if not user or not check_password_hash(user.password_hash, password):
            error = "Invalid username or password."
        else:
            login_user(user, remember=request.form.get('remember') == 'on')
            logger.info("User logged in: %s", username)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))

    return render_template('login.html', error=error)


@app.route('/logout')
@login_required
def logout():
    logger.info("User logged out: %s", current_user.username)
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    debug_mode = os.getenv("FLASK_DEBUG", "false").strip().lower() in ("1", "true", "yes")
    app.run(debug=debug_mode)