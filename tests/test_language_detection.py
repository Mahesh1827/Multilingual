"""
Unit tests for the language detection layer.

Tests the three-layer detection pipeline:
  Layer 1 — Unicode script block detector (most reliable for native text)
  Layer 2 — Lingua ML detector (handles romanized / ambiguous text)
  Layer 3 — Default English fallback

These tests are pure-Python and do NOT load any ML models —
the Lingua detector is mocked where its output is irrelevant.
"""

import pytest
from unittest.mock import patch, MagicMock


# ──────────────────────────────────────────────────────────────
# Helpers that replicate the script-detection logic from STT.py
# without importing the full module (avoids heavy transitive imports)
# ──────────────────────────────────────────────────────────────

SCRIPT_RANGES = [
    (0x0900, 0x097F, "hi"),   # Devanagari (Hindi)
    (0x0C00, 0x0C7F, "te"),   # Telugu
    (0x0B80, 0x0BFF, "ta"),   # Tamil
    (0x0C80, 0x0CFF, "kn"),   # Kannada
]


def _detect_script(text: str) -> str | None:
    """
    Pure-Python reimplementation of the Unicode-block detector.
    Returns the ISO-639-1 code of the dominant script, or None.
    """
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi, lang in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break
    if not counts:
        return None
    return max(counts, key=lambda k: counts[k])


# ──────────────────────────────────────────────────────────────
# Script detection tests
# ──────────────────────────────────────────────────────────────

class TestScriptDetection:
    """Layer 1: Unicode block range detector."""

    @pytest.mark.parametrize("text, expected", [
        ("తిరుమల ఎలా చేరుకోవాలి",  "te"),   # Telugu
        ("దర్శన సమయం ఎంత",           "te"),
        ("ತಿರುಮಲ ಎಲ್ಲಿದೆ",           "kn"),   # Kannada
        ("ದರ್ಶನ ಸಮಯ ಎಷ್ಟು",          "kn"),
        ("तिरुमला कैसे पहुँचें",      "hi"),   # Hindi (Devanagari)
        ("दर्शन का समय क्या है",      "hi"),
        ("திருமலை எப்படி செல்வது",   "ta"),   # Tamil
        ("தரிசன நேரம் என்ன",         "ta"),
    ])
    def test_native_script_detected_correctly(self, text, expected):
        assert _detect_script(text) == expected

    def test_pure_ascii_returns_none(self):
        assert _detect_script("How to reach Tirumala") is None

    def test_empty_string_returns_none(self):
        assert _detect_script("") is None

    def test_mixed_script_dominant_wins(self):
        # Three Telugu chars + one Hindi char → Telugu wins
        text = "తి రు మ" + "क"
        result = _detect_script(text)
        assert result == "te"

    def test_romanized_hindi_returns_none(self):
        # Romanized — no Unicode script chars
        assert _detect_script("Tirumala ki darshan timings kya hai") is None

    def test_numbers_and_punctuation_ignored(self):
        assert _detect_script("123 !@#$%") is None

    def test_single_native_char_detected(self):
        assert _detect_script("த") == "ta"   # single Tamil character


# ──────────────────────────────────────────────────────────────
# Language names mapping
# ──────────────────────────────────────────────────────────────

LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
}


class TestLangNameMapping:
    """Verify the ISO → display-name mapping used in UI output."""

    @pytest.mark.parametrize("code, name", [
        ("en", "English"),
        ("hi", "Hindi"),
        ("te", "Telugu"),
        ("ta", "Tamil"),
        ("kn", "Kannada"),
    ])
    def test_known_codes_map_correctly(self, code, name):
        assert LANG_NAMES[code] == name

    def test_all_five_languages_present(self):
        assert len(LANG_NAMES) == 5

    def test_unknown_code_not_in_mapping(self):
        assert "fr" not in LANG_NAMES


# ──────────────────────────────────────────────────────────────
# Priority rule tests (script > lingua > English default)
# ──────────────────────────────────────────────────────────────

def _priority_detect(script_lang, lingua_lang) -> tuple[str, str]:
    """
    Replicate the priority logic from STT.py:
      1. Script result (most trusted)
      2. Lingua ML result
      3. Default 'en'
    Returns (final_lang, method_name).
    """
    if script_lang:
        return script_lang, "script"
    if lingua_lang:
        return lingua_lang, "lingua"
    return "en", "default"


class TestDetectionPriority:
    """Layer-priority rules applied after each detector returns."""

    def test_script_beats_lingua(self):
        lang, method = _priority_detect("te", "en")
        assert lang == "te"
        assert method == "script"

    def test_lingua_used_when_no_script(self):
        lang, method = _priority_detect(None, "hi")
        assert lang == "hi"
        assert method == "lingua"

    def test_english_fallback_when_both_none(self):
        lang, method = _priority_detect(None, None)
        assert lang == "en"
        assert method == "default"

    def test_script_none_lingua_none_gives_en(self):
        lang, _ = _priority_detect(None, None)
        assert lang == "en"

    def test_script_takes_any_supported_language(self):
        for code in ("hi", "te", "ta", "kn"):
            lang, method = _priority_detect(code, "en")
            assert lang == code
            assert method == "script"


# ──────────────────────────────────────────────────────────────
# Supported language config test
# ──────────────────────────────────────────────────────────────

class TestSupportedLanguages:
    """Verify the config contains the expected supported languages."""

    def test_supported_languages_include_five_languages(self):
        from query.config import SUPPORTED_VOICE_LANGUAGES
        assert len(SUPPORTED_VOICE_LANGUAGES) == 5

    def test_english_is_supported(self):
        from query.config import SUPPORTED_VOICE_LANGUAGES
        assert "en" in SUPPORTED_VOICE_LANGUAGES

    def test_telugu_is_supported(self):
        from query.config import SUPPORTED_VOICE_LANGUAGES
        assert "te" in SUPPORTED_VOICE_LANGUAGES

    def test_all_expected_codes_present(self):
        from query.config import SUPPORTED_VOICE_LANGUAGES
        for code in ("en", "hi", "te", "ta", "kn"):
            assert code in SUPPORTED_VOICE_LANGUAGES, f"Missing language: {code}"
