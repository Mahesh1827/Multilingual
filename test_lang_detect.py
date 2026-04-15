"""
Language Detection Test Tool
=============================
Test the full language detection stack without a microphone.
Type any text (native script or romanized) and see exactly
which detector fired and what language was identified.

Usage:
    python test_lang_detect.py
    python test_lang_detect.py --text "ತಿರುಮಲ ಎಲ್ಲಿದೆ"
    python test_lang_detect.py --batch          # runs the built-in test suite
"""

import sys
import os
import argparse

# Force UTF-8 output on Windows so native-script characters print correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────
# Import detection functions directly from STT
# ──────────────────────────────────────────────
from query.STT import (
    _detect_script_language,
    _detect_language_lingua,
    _correct_language,
    _LINGUA_TO_ISO,
)

LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
}

# ANSI colours for terminal output
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _color(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}"


# ──────────────────────────────────────────────
# Core detection logic (mirrors STT.py pipeline)
# ──────────────────────────────────────────────

def detect_language(text: str) -> dict:
    """
    Run all three detection layers and return a detailed report.
    This mirrors the exact priority order used in STT.py.
    """
    result = {
        "text":        text,
        "script":      None,   # Unicode block detector
        "lingua":      None,   # Lingua ML detector
        "final":       None,   # Winner after priority rules
        "method":      None,   # Which detector won
        "script_counts": {},   # per-language script char counts
    }

    # ── Layer 1: Script (Unicode block ranges) ──────────────────────────────
    # Counts characters per script block; most reliable for native text.
    script_counts: dict[str, int] = {}
    total_non_ascii = 0
    SCRIPT_RANGES = [
        (0x0900, 0x097F, "hi"),
        (0x0C00, 0x0C7F, "te"),
        (0x0B80, 0x0BFF, "ta"),
        (0x0C80, 0x0CFF, "kn"),
    ]
    for ch in text:
        cp = ord(ch)
        if cp > 127:
            total_non_ascii += 1
        for lo, hi, lang in SCRIPT_RANGES:
            if lo <= cp <= hi:
                script_counts[lang] = script_counts.get(lang, 0) + 1
                break

    result["script_counts"] = {
        LANG_NAMES.get(k, k): v for k, v in script_counts.items()
    }

    script_lang = _detect_script_language(text)
    result["script"] = script_lang

    # ── Layer 2: Lingua ML ──────────────────────────────────────────────────
    lingua_lang = _detect_language_lingua(text)
    result["lingua"] = lingua_lang

    # ── Priority: script > lingua > "en" default ─────────────────────────────
    if script_lang:
        result["final"]  = script_lang
        result["method"] = "script (Unicode block ranges)"
    elif lingua_lang:
        result["final"]  = lingua_lang
        result["method"] = "lingua (ML)"
    else:
        result["final"]  = "en"
        result["method"] = "default fallback"

    return result


# ──────────────────────────────────────────────
# Pretty-print
# ──────────────────────────────────────────────

def _print_result(r: dict, expected: str | None = None) -> bool:
    """Print detection result. Returns True if correct (when expected is given)."""
    lang    = r["final"]
    name    = LANG_NAMES.get(lang, lang.upper())
    method  = r["method"]
    correct = (expected is None) or (lang == expected)

    status = ""
    if expected is not None:
        exp_name = LANG_NAMES.get(expected, expected.upper())
        if correct:
            status = _color(f"  ✓  PASS  (expected {exp_name})", _GREEN)
        else:
            status = _color(f"  ✗  FAIL  (expected {exp_name}, got {name})", _RED)

    print(f"\n  Text    : {_color(r['text'], _CYAN)}")
    if r["script_counts"]:
        counts_str = ", ".join(f"{k}:{v}" for k, v in r["script_counts"].items())
        print(f"  Script  : {r['script'] and LANG_NAMES.get(r['script'], r['script']) or _color('none', _YELLOW)}  [{counts_str}]")
    else:
        print(f"  Script  : {_color('no native chars (ASCII)', _YELLOW)}")
    print(f"  Lingua  : {r['lingua'] and LANG_NAMES.get(r['lingua'], r['lingua']) or _color('none', _YELLOW)}")
    print(f"  {_BOLD}Final   : {_color(name, _GREEN if correct else _RED)}  via {method}{_RESET}{status}")
    return correct


# ──────────────────────────────────────────────
# Built-in test suite
# ──────────────────────────────────────────────

_TEST_CASES = [
    # (text, expected_lang, description)

    # ── Telugu ──
    ("తిరుమల ఎలా చేరుకోవాలి", "te", "Telugu — how to reach Tirumala"),
    ("దర్శన సమయం ఎంత", "te", "Telugu — darshan timings"),
    ("ఏడు కొండలు పేర్లు ఏమిటి", "te", "Telugu — names of 7 hills"),
    ("లడ్డూ ధర ఎంత", "te", "Telugu — laddu price"),

    # ── Kannada ──
    ("ತಿರುಮಲ ಎಲ್ಲಿದೆ", "kn", "Kannada — where is Tirumala"),
    ("ದರ್ಶನ ಸಮಯ ಎಷ್ಟು", "kn", "Kannada — darshan timings"),
    ("ಏಳು ಬೆಟ್ಟಗಳ ಹೆಸರುಗಳು", "kn", "Kannada — names of 7 hills"),
    ("ಲಡ್ಡು ಬೆಲೆ ಎಷ್ಟು", "kn", "Kannada — laddu price"),

    # ── Hindi ──
    ("तिरुमला कैसे पहुँचें", "hi", "Hindi — how to reach Tirumala"),
    ("दर्शन का समय क्या है", "hi", "Hindi — darshan timings"),
    ("सात पहाड़ियों के नाम", "hi", "Hindi — names of 7 hills"),

    # ── Tamil ──
    ("திருமலை எப்படி செல்வது", "ta", "Tamil — how to reach Tirumala"),
    ("தரிசன நேரம் என்ன", "ta", "Tamil — darshan timings"),

    # ── English ──
    ("How to reach Tirumala from Tirupati", "en", "English — plain question"),
    ("What are the darshan timings at Tirumala", "en", "English — darshan timings"),
    ("Tirumala TTD laddu cost", "en", "English — short query"),

    # ── Edge cases ──
    ("Tirumala ki darshan timings kya hai", "en", "Romanized Hindi (no script chars)"),
    ("Tirumala ki darshan", "en", "Mixed romanized"),
]


def run_batch():
    print(_color("\n=== Language Detection Test Suite ===\n", _BOLD))
    passed = 0
    failed = 0
    failures = []

    for text, expected, desc in _TEST_CASES:
        print(_color(f"[{desc}]", _BOLD))
        r = detect_language(text)
        ok = _print_result(r, expected)
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append((desc, expected, r["final"]))

    print("\n" + "─" * 50)
    total = passed + failed
    print(_color(f"  Results: {passed}/{total} passed", _GREEN if failed == 0 else _YELLOW))
    if failures:
        print(_color("\n  Failed cases:", _RED))
        for desc, exp, got in failures:
            exp_name = LANG_NAMES.get(exp, exp)
            got_name = LANG_NAMES.get(got, got)
            print(f"    • {desc}: expected {exp_name}, got {got_name}")
    print()


# ──────────────────────────────────────────────
# Interactive loop
# ──────────────────────────────────────────────

def run_interactive():
    print(_color("\n=== Language Detection — Interactive Mode ===", _BOLD))
    print("Type any text to detect its language.")
    print("Commands:  'batch' → run full test suite   |   'quit' or Ctrl+C → exit\n")

    while True:
        try:
            text = input("Enter text: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            break
        if text.lower() == "batch":
            run_batch()
            continue

        r = detect_language(text)
        _print_result(r)
        print()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test language detection")
    parser.add_argument("--text",  type=str, help="Detect language for a single text string")
    parser.add_argument("--batch", action="store_true", help="Run the built-in test suite")
    args = parser.parse_args()

    if args.text:
        r = detect_language(args.text)
        _print_result(r)
        print()
    elif args.batch:
        run_batch()
    else:
        run_interactive()
