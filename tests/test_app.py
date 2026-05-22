"""
Comprehensive unit tests for the Email Security System.

Covers:
- Text preprocessing
- URL extraction and feature engineering
- Domain parsing utilities
- Brand impersonation detection
- Rule-based phishing scoring (full pipeline)

Run with:
    pytest tests/test_app.py -v
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make the parent directory importable so we can `import app`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Patch joblib.load BEFORE importing app, since app.py loads models at import time
with patch("joblib.load") as mock_load:
    mock_load.return_value = None
    import app


# ---------------------------------------------------------------------------
# preprocess_text
# ---------------------------------------------------------------------------
class TestPreprocessText:
    def test_lowercases_text(self):
        assert app.preprocess_text("HELLO World") == "hello world"

    def test_removes_urls(self):
        result = app.preprocess_text("Click https://bad.com/login now")
        assert "https" not in result
        assert "bad.com" not in result
        assert "click" in result
        assert "now" in result

    def test_removes_www_urls(self):
        result = app.preprocess_text("visit www.phish.tk today")
        assert "www" not in result
        assert "phish.tk" not in result

    def test_removes_special_characters(self):
        result = app.preprocess_text("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_collapses_whitespace(self):
        assert app.preprocess_text("hello    world") == "hello world"

    def test_empty_string(self):
        assert app.preprocess_text("") == ""

    def test_keeps_alphanumeric(self):
        assert app.preprocess_text("account 123") == "account 123"


# ---------------------------------------------------------------------------
# extract_urls
# ---------------------------------------------------------------------------
class TestExtractUrls:
    def test_extracts_http_url(self):
        urls = app.extract_urls("visit http://example.com today")
        assert urls == ["http://example.com"]

    def test_extracts_https_url(self):
        urls = app.extract_urls("go to https://secure.bank.com/login")
        assert urls == ["https://secure.bank.com/login"]

    def test_extracts_www_url(self):
        urls = app.extract_urls("visit www.example.com")
        assert urls == ["www.example.com"]

    def test_extracts_multiple_urls(self):
        text = "http://a.com and https://b.com and www.c.com"
        urls = app.extract_urls(text)
        assert len(urls) == 3

    def test_no_urls(self):
        assert app.extract_urls("Just a normal email body.") == []

    def test_empty_string(self):
        assert app.extract_urls("") == []


# ---------------------------------------------------------------------------
# extract_url_features
# ---------------------------------------------------------------------------
class TestExtractUrlFeatures:
    def test_no_urls_returns_zeros(self):
        features = app.extract_url_features("no urls here")
        assert features["url_count"] == 0
        assert features["max_url_length"] == 0
        assert features["avg_url_length"] == 0.0
        assert features["has_ip_url"] == 0
        assert features["uses_https"] == 0

    def test_counts_urls(self):
        features = app.extract_url_features("http://a.com http://b.com http://c.com")
        assert features["url_count"] == 3

    def test_detects_ip_url(self):
        features = app.extract_url_features("http://192.168.1.1/login")
        assert features["has_ip_url"] == 1

    def test_detects_at_symbol(self):
        features = app.extract_url_features("http://user@evil.com")
        assert features["has_at_symbol"] == 1

    def test_detects_dash(self):
        features = app.extract_url_features("http://secure-paypal.com")
        assert features["has_dash"] == 1

    def test_detects_many_dots(self):
        features = app.extract_url_features("http://a.b.c.d.evil.com")
        assert features["has_many_dots"] == 1

    def test_detects_suspicious_pattern(self):
        features = app.extract_url_features("http://example.com/verify/login")
        assert features["has_suspicious_pattern"] == 1

    def test_detects_https(self):
        features = app.extract_url_features("https://example.com")
        assert features["uses_https"] == 1

    def test_http_only_not_https(self):
        features = app.extract_url_features("http://example.com")
        assert features["uses_https"] == 0

    def test_avg_url_length(self):
        features = app.extract_url_features("http://a.com http://bbbbbbbb.com")
        assert features["avg_url_length"] == pytest.approx(
            (len("http://a.com") + len("http://bbbbbbbb.com")) / 2
        )


# ---------------------------------------------------------------------------
# Domain parsing
# ---------------------------------------------------------------------------
class TestGetSenderDomain:
    def test_extracts_domain(self):
        assert app.get_sender_domain("user@example.com") == "example.com"

    def test_lowercases(self):
        assert app.get_sender_domain("User@Example.COM") == "example.com"

    def test_no_at_symbol_returns_empty(self):
        assert app.get_sender_domain("notanemail") == ""

    def test_empty_input(self):
        assert app.get_sender_domain("") == ""

    def test_strips_whitespace(self):
        assert app.get_sender_domain("  user@example.com  ") == "example.com"


class TestGetUrlDomain:
    def test_http_url(self):
        assert app.get_url_domain("http://example.com/path") == "example.com"

    def test_https_url(self):
        assert app.get_url_domain("https://example.com/path") == "example.com"

    def test_bare_domain_gets_scheme(self):
        assert app.get_url_domain("www.example.com/path") == "www.example.com"

    def test_empty_input(self):
        assert app.get_url_domain("") == ""


class TestGetSenderLocalPart:
    def test_extracts_local(self):
        assert app.get_sender_local_part("john.doe@example.com") == "john.doe"

    def test_lowercases(self):
        assert app.get_sender_local_part("Support@Example.com") == "support"

    def test_no_at_returns_input(self):
        assert app.get_sender_local_part("nobody") == "nobody"


class TestIsFreeEmailDomain:
    @pytest.mark.parametrize("domain", [
        "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
        "icloud.com", "proton.me",
    ])
    def test_free_domains(self, domain):
        assert app.is_free_email_domain(domain) is True

    @pytest.mark.parametrize("domain", [
        "paypal.com", "mycompany.org", "bank.co.uk",
    ])
    def test_non_free_domains(self, domain):
        assert app.is_free_email_domain(domain) is False


class TestGetRegisteredLikeDomain:
    def test_simple_domain(self):
        assert app.get_registered_like_domain("example.com") == "example.com"

    def test_strips_subdomain(self):
        assert app.get_registered_like_domain("mail.example.com") == "example.com"

    def test_strips_deep_subdomain(self):
        assert (
            app.get_registered_like_domain("login.secure.paypal.com")
            == "paypal.com"
        )

    def test_strips_port(self):
        assert app.get_registered_like_domain("example.com:8080") == "example.com"

    def test_empty_input(self):
        assert app.get_registered_like_domain("") == ""


# ---------------------------------------------------------------------------
# count_subdomains — the function we just fixed!
# ---------------------------------------------------------------------------
class TestCountSubdomains:
    def test_no_subdomains(self):
        assert app.count_subdomains("example.com") == 0

    def test_one_subdomain(self):
        assert app.count_subdomains("mail.example.com") == 1

    def test_two_subdomains(self):
        assert app.count_subdomains("login.mail.example.com") == 2

    def test_three_subdomains(self):
        # Classic phishing pattern: login.secure.verify.paypal.com
        assert app.count_subdomains("login.secure.verify.paypal.com") == 3

    def test_empty_string(self):
        assert app.count_subdomains("") == 0

    def test_none_safe(self):
        assert app.count_subdomains(None) == 0

    def test_strips_port(self):
        assert app.count_subdomains("example.com:8080") == 0

    def test_uppercase_normalized(self):
        assert app.count_subdomains("Mail.Example.COM") == 1


# ---------------------------------------------------------------------------
# Brand and domain abuse detection
# ---------------------------------------------------------------------------
class TestContainsBrandAndExtraWords:
    def test_detects_paypal_with_secure(self):
        result, desc = app.contains_brand_and_extra_words(
            "secure-paypal.com", ["paypal", "apple"]
        )
        assert result is True
        assert "paypal" in desc

    def test_clean_brand_domain_not_flagged(self):
        # Just "paypal.com" alone — no extra suspicious words
        result, _ = app.contains_brand_and_extra_words("paypal.com", ["paypal"])
        assert result is False

    def test_no_brand_match(self):
        result, _ = app.contains_brand_and_extra_words(
            "mycompany.com", ["paypal", "apple"]
        )
        assert result is False

    def test_detects_verify_keyword(self):
        result, desc = app.contains_brand_and_extra_words(
            "paypal-verify.com", ["paypal"]
        )
        assert result is True
        assert "verify" in desc


class TestHasMisleadingSubdomain:
    def test_brand_in_subdomain_only(self):
        # Brand appears as subdomain but registered domain is evil.com
        result, brand = app.has_misleading_subdomain(
            "paypal.evil.com", ["paypal"]
        )
        assert result is True
        assert brand == "paypal"

    def test_legitimate_brand_domain(self):
        # Legitimate paypal.com — brand is in the registered domain
        result, _ = app.has_misleading_subdomain("paypal.com", ["paypal"])
        assert result is False

    def test_no_brand_at_all(self):
        result, _ = app.has_misleading_subdomain("random.com", ["paypal"])
        assert result is False


class TestIsLongDomain:
    def test_short_domain(self):
        assert app.is_long_domain("example.com") is False

    def test_long_domain(self):
        assert app.is_long_domain("a" * 30 + ".com") is True

    def test_boundary_below_threshold(self):
        # 29 chars = below threshold, should be False
        assert app.is_long_domain("a" * 25 + ".com") is False

    def test_boundary_at_threshold(self):
        # 30 chars = at threshold (>= 30), should be True
        assert app.is_long_domain("a" * 26 + ".com") is True

    def test_empty_domain(self):
        assert app.is_long_domain("") is False


# ---------------------------------------------------------------------------
# contains_any helper
# ---------------------------------------------------------------------------
class TestContainsAny:
    def test_finds_keyword(self):
        assert app.contains_any("verify your account", ["verify", "login"]) == ["verify"]

    def test_finds_multiple(self):
        hits = app.contains_any("verify and login now", ["verify", "login"])
        assert "verify" in hits
        assert "login" in hits

    def test_case_insensitive(self):
        assert app.contains_any("VERIFY NOW", ["verify"]) == ["verify"]

    def test_no_match(self):
        assert app.contains_any("hello world", ["verify"]) == []


# ---------------------------------------------------------------------------
# Rule-based phishing boost (full pipeline)
# ---------------------------------------------------------------------------
class TestApplyRuleBasedPhishingBoost:
    def test_clean_email_low_score(self):
        label, conf, score, reasons = app.apply_rule_based_phishing_boost(
            sender="friend@example.com",
            subject="Lunch tomorrow?",
            body="Hey, want to grab lunch tomorrow around noon?",
            urls=[],
            ml_label="Safe",
            ml_confidence=95.0,
        )
        assert label == "Safe"
        assert score < 5

    def test_obvious_phishing_boosts_to_high_risk(self):
        label, conf, score, reasons = app.apply_rule_based_phishing_boost(
            sender="support@secure-paypal-verify.tk",
            subject="URGENT: Verify your account NOW",
            body=(
                "Your account has been suspended. Click here immediately "
                "to verify: http://192.168.1.1/paypal-login/verify"
            ),
            urls=["http://192.168.1.1/paypal-login/verify"],
            ml_label="Spam/Phishing",
            ml_confidence=85.0,
        )
        assert label == "High Risk Phishing"
        assert score >= 10

    def test_ip_url_adds_points(self):
        _, _, score, reasons = app.apply_rule_based_phishing_boost(
            sender="x@x.com",
            subject="hi",
            body="click http://192.168.1.1/page",
            urls=["http://192.168.1.1/page"],
            ml_label="Safe",
            ml_confidence=90.0,
        )
        assert any("IP" in r for r in reasons)

    def test_url_shortener_detected(self):
        _, _, score, reasons = app.apply_rule_based_phishing_boost(
            sender="x@x.com",
            subject="hi",
            body="click http://bit.ly/abc123",
            urls=["http://bit.ly/abc123"],
            ml_label="Safe",
            ml_confidence=90.0,
        )
        assert any("shortening" in r.lower() for r in reasons)

    def test_brand_with_free_email_flagged(self):
        # PayPal-themed email from a gmail.com address — classic phishing
        _, _, score, reasons = app.apply_rule_based_phishing_boost(
            sender="paypal.support@gmail.com",
            subject="PayPal account issue",
            body="Verify your paypal account now.",
            urls=[],
            ml_label="Safe",
            ml_confidence=90.0,
        )
        assert any("free email" in r.lower() for r in reasons)

    def test_suspicious_tld_flagged(self):
        _, _, score, reasons = app.apply_rule_based_phishing_boost(
            sender="x@phish.tk",
            subject="hello",
            body="click http://phish.tk/login",
            urls=["http://phish.tk/login"],
            ml_label="Safe",
            ml_confidence=90.0,
        )
        assert any(".tk" in r or "top-level" in r.lower() for r in reasons)

    def test_misleading_subdomain_flagged(self):
        # paypal.com appears as subdomain, but registered domain is evil.com
        _, _, score, reasons = app.apply_rule_based_phishing_boost(
            sender="x@x.com",
            subject="update",
            body="go to https://paypal.com.evil.com/login",
            urls=["https://paypal.com.evil.com/login"],
            ml_label="Safe",
            ml_confidence=90.0,
        )
        assert any("misleading" in r.lower() or "subdomain" in r.lower() for r in reasons)

    def test_ml_safe_becomes_suspicious(self):
        # Moderate signals should push a "Safe" ML label to "Suspicious"
        label, _, score, _ = app.apply_rule_based_phishing_boost(
            sender="support@login-secure.xyz",
            subject="Verify your account",
            body="Please click here to login and verify immediately.",
            urls=["http://bit.ly/xyz"],
            ml_label="Safe",
            ml_confidence=80.0,
        )
        # Depending on exact scoring, should end up Suspicious or High Risk
        assert label in ("Suspicious", "High Risk Phishing")
        assert score >= 5

    def test_virustotal_boost_applied(self):
        vt_summary = {
            "checked": True,
            "status": "OK",
            "total_score": 8,  # capped at +6
            "results": [
                {"url": "http://bad.com", "malicious": 4, "suspicious": 0},
            ],
        }
        _, _, score, reasons = app.apply_rule_based_phishing_boost(
            sender="x@x.com",
            subject="hi",
            body="visit http://bad.com",
            urls=["http://bad.com"],
            ml_label="Safe",
            ml_confidence=90.0,
            vt_summary=vt_summary,
        )
        assert any("VirusTotal" in r for r in reasons)
        assert any("malicious=4" in r for r in reasons)

    def test_empty_inputs_dont_crash(self):
        label, conf, score, reasons = app.apply_rule_based_phishing_boost(
            sender="",
            subject="",
            body="",
            urls=[],
            ml_label="Safe",
            ml_confidence=50.0,
        )
        assert label == "Safe"
        assert isinstance(reasons, list)


# ---------------------------------------------------------------------------
# VirusTotal URL ID encoding
# ---------------------------------------------------------------------------
class TestUrlToVtId:
    def test_returns_string(self):
        result = app.url_to_vt_id("http://example.com")
        assert isinstance(result, str)

    def test_strips_padding(self):
        result = app.url_to_vt_id("http://example.com")
        assert not result.endswith("=")

    def test_different_urls_different_ids(self):
        assert app.url_to_vt_id("http://a.com") != app.url_to_vt_id("http://b.com")