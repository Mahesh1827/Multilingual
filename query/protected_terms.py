"""
protected_terms.py
==================
Prevents domain-specific Tirumala / TTD proper nouns from being
incorrectly translated by Google Translate.

Problem it solves
-----------------
Google Translate is linguistically correct but context-unaware:
  "दिव्य दर्शन" → "divine darshan"   (WRONG — it's a ticket category name)
  "सेवा"         → "service"          (WRONG — it's a religious rite name)
With this module:
  "दिव्य दर्शन" → "Divya Darshan"    ✓
  "सेवा"         → "Seva"             ✓

How it works
------------
1. mask_protected_terms(text, lang)
      Scans the source text for known domain terms and replaces them with
      unique opaque placeholders like __TTERM_0__ before translation.
      Returns (masked_text, restore_map).

2. restore_protected_terms(translated, restore_map)
      After Google Translate runs, restores each placeholder with the
      canonical English / transliterated name.

Adding new terms
----------------
Add entries to the appropriate _XX_TERMS list below.
IMPORTANT: list longer / more-specific phrases BEFORE shorter sub-phrases
           so "दिव्य दर्शन" is matched before "दर्शन".
"""

import logging
import re

logger = logging.getLogger(__name__)

_PLACEHOLDER_PREFIX = "__TTERM_"
_PLACEHOLDER_SUFFIX = "__"

# ─────────────────────────────────────────────────────────────────────────────
# TERM REGISTRY  (native script → canonical English/transliterated name)
# ─────────────────────────────────────────────────────────────────────────────
# Rules:
#   • Longer / more-specific phrases MUST appear before shorter sub-phrases.
#   • Include common spelling/ASR variants where possible.
# ─────────────────────────────────────────────────────────────────────────────

# ── Hindi (Devanagari script) ─────────────────────────────────────────────────
_HI_TERMS: list[tuple[str, str]] = [

    # Darshan ticket types  (most specific first)
    ("दिव्य दर्शन",                     "Divya Darshan"),
    ("दिव्यदर्शन",                       "Divya Darshan"),
    ("सर्व दर्शन",                       "Sarva Darshan"),
    ("सर्वदर्शन",                         "Sarva Darshan"),
    ("विशेष दर्शन",                      "Vishesh Darshan"),
    ("स्लॉटेड दर्शन",                    "Slotted Darshan"),
    ("प्राथमिकता दर्शन",                 "Priority Darshan"),
    ("वीआईपी दर्शन",                     "VIP Darshan"),

    # Seva types  (temple rituals)
    ("सुप्रभातम् सेवा",                  "Suprabhatam Seva"),
    ("सुप्रभातम्",                        "Suprabhatam"),
    ("सुप्रभातम",                         "Suprabhatam"),
    ("थोमाल सेवा",                       "Thomala Seva"),
    ("अर्जित सेवा",                      "Arjita Seva"),
    ("अर्जित ब्रह्मोत्सवम्",            "Arjita Brahmotsavam"),
    ("अर्चना सेवा",                      "Archana Seva"),
    ("निजपाद दर्शनम्",                  "Nijapada Darshanam"),
    ("अष्टदल पाद पद्माराधना",           "Astadala Pada Padmaradhana"),
    ("चक्रस्नानम्",                      "Chakrasnanam"),
    ("ऊंजल सेवा",                        "Unjal Seva"),
    ("सहस्र दीपालंकरण सेवा",            "Sahasra Deepalankara Seva"),
    ("कल्याणोत्सवम्",                    "Kalyanaotsavam"),
    ("कल्याणोत्सव",                      "Kalyanaotsavam"),
    ("पुष्पयागम्",                       "Pushpayagam"),

    # Prasadam / offerings
    ("लड्डू प्रसादम्",                   "Laddu Prasadam"),
    ("लड्डू प्रसाद",                     "Laddu Prasadam"),
    ("अन्न प्रसादम्",                    "Anna Prasadam"),
    ("अन्न प्रसाद",                      "Anna Prasadam"),
    ("प्रसादम्",                          "Prasadam"),
    ("प्रसाद",                            "Prasadam"),
    ("लड्डू",                             "Laddu"),
    ("लडू",                               "Laddu"),

    # Festivals / events
    ("ब्रह्मोत्सवम्",                    "Brahmotsavam"),
    ("ब्रह्मोत्सव",                       "Brahmotsavam"),
    ("वैकुंठ एकादशी",                    "Vaikunta Ekadasi"),
    ("वैकुण्ठ एकादशी",                   "Vaikunta Ekadasi"),
    ("रथोत्सव",                           "Rathotsavam"),

    # Places / organisation
    ("तिरुमला तिरुपति देवस्थानम्",      "Tirumala Tirupati Devasthanams"),
    ("तिरुमला तिरुपती देवस्थानम",       "Tirumala Tirupati Devasthanams"),
    ("तिरुमला",                           "Tirumala"),
    ("तिरुपति",                           "Tirupati"),
    ("तिरुपती",                           "Tirupati"),
    ("वेंकटेश्वर",                        "Venkateswara"),
    ("वेंकटेस्वर",                        "Venkateswara"),
    ("बालाजी",                            "Balaji"),
    ("वेंकटचलपति",                        "Venkatchalapati"),
    ("शेषाचलम्",                          "Seshachalam"),

    # General TTD temple terms left in native form by convention
    ("दर्शन",                             "Darshan"),
    ("सेवा",                              "Seva"),
    ("हुंडी",                             "Hundi"),
    ("गोपुरम्",                           "Gopuram"),
    ("गोपुरम",                            "Gopuram"),
    ("विमानम्",                           "Vimanam"),
    ("मंडपम्",                            "Mandapam"),
    ("आगम",                               "Agama"),
    ("ध्वजस्तंभ",                         "Dhwajastambham"),
]

# ── Telugu (Telugu script) ────────────────────────────────────────────────────
_TE_TERMS: list[tuple[str, str]] = [

    # Darshan ticket types
    ("దివ్య దర్శనం",                     "Divya Darshan"),
    ("దివ్య దర్శన",                      "Divya Darshan"),
    ("సర్వ దర్శనం",                      "Sarva Darshan"),
    ("సర్వ దర్శన",                       "Sarva Darshan"),
    ("విశేష దర్శనం",                     "Vishesh Darshan"),
    ("స్లాటెడ్ దర్శనం",                 "Slotted Darshan"),
    ("విఐపి దర్శనం",                     "VIP Darshan"),

    # Seva types
    ("సుప్రభాతం సేవ",                    "Suprabhatam Seva"),
    ("సుప్రభాతం",                        "Suprabhatam"),
    ("థోమాల సేవ",                        "Thomala Seva"),
    ("ఆర్జిత సేవ",                       "Arjita Seva"),
    ("అర్చన సేవ",                        "Archana Seva"),
    ("నిజపాద దర్శనం",                    "Nijapada Darshanam"),
    ("కల్యాణోత్సవం",                     "Kalyanaotsavam"),
    ("పుష్పయాగం",                        "Pushpayagam"),
    ("సహస్ర దీపాలంకరణ సేవ",            "Sahasra Deepalankara Seva"),

    # Prasadam
    ("లడ్డు ప్రసాదం",                    "Laddu Prasadam"),
    ("అన్న ప్రసాదం",                     "Anna Prasadam"),
    ("ప్రసాదం",                           "Prasadam"),
    ("లడ్డు",                             "Laddu"),
    ("లడ్డూ",                             "Laddu"),

    # Festivals / events
    ("బ్రహ్మోత్సవాలు",                   "Brahmotsavam"),
    ("బ్రహ్మోత్సవం",                     "Brahmotsavam"),
    ("వైకుంఠ ఏకాదశి",                   "Vaikunta Ekadasi"),
    ("రథోత్సవం",                         "Rathotsavam"),

    # Places / org
    ("తిరుమల తిరుపతి దేవస్థానాలు",     "Tirumala Tirupati Devasthanams"),
    ("తిరుమల",                            "Tirumala"),
    ("తిరుపతి",                           "Tirupati"),
    ("వేంకటేశ్వర",                        "Venkateswara"),
    ("బాలాజీ",                            "Balaji"),
    ("శేషాచలం",                           "Seshachalam"),

    # General terms
    ("దర్శనం",                            "Darshan"),
    ("దర్శన",                             "Darshan"),
    ("సేవ",                               "Seva"),
    ("హుండీ",                             "Hundi"),
    ("గోపురం",                            "Gopuram"),
    ("విమానం",                            "Vimanam"),
    ("మండపం",                             "Mandapam"),
]

# ── Tamil (Tamil script) ──────────────────────────────────────────────────────
_TA_TERMS: list[tuple[str, str]] = [

    # Darshan ticket types
    ("திவ்ய தரிசனம்",                    "Divya Darshan"),
    ("திவ்ய தரிசன",                      "Divya Darshan"),
    ("சர்வ தரிசனம்",                     "Sarva Darshan"),
    ("விஷேஷ தரிசனம்",                    "Vishesh Darshan"),
    ("விஐபி தரிசனம்",                    "VIP Darshan"),

    # Seva types
    ("சுப்ரபாதம் சேவை",                  "Suprabhatam Seva"),
    ("சுப்ரபாதம்",                        "Suprabhatam"),
    ("தொமால சேவை",                       "Thomala Seva"),
    ("அர்ஜித சேவை",                      "Arjita Seva"),
    ("கல்யாணோத்சவம்",                    "Kalyanaotsavam"),

    # Prasadam
    ("லட்டு பிரசாதம்",                   "Laddu Prasadam"),
    ("அன்ன பிரசாதம்",                    "Anna Prasadam"),
    ("பிரசாதம்",                          "Prasadam"),
    ("லட்டு",                             "Laddu"),

    # Festivals / events
    ("பிரம்மோத்சவம்",                    "Brahmotsavam"),
    ("வைகுண்ட ஏகாதசி",                  "Vaikunta Ekadasi"),

    # Places / org
    ("திருமலை திருப்பதி தேவஸ்தானம்",   "Tirumala Tirupati Devasthanams"),
    ("திருமலை",                           "Tirumala"),
    ("திருப்பதி",                         "Tirupati"),
    ("வேங்கடேஸ்வரர்",                    "Venkateswara"),
    ("வேங்கடேஸ்வர",                      "Venkateswara"),
    ("பாலாஜி",                            "Balaji"),

    # General terms
    ("தரிசனம்",                           "Darshan"),
    ("சேவை",                              "Seva"),
    ("குண்டி",                            "Hundi"),
    ("கோபுரம்",                           "Gopuram"),
]

# ── Kannada (Kannada script) ──────────────────────────────────────────────────
_KN_TERMS: list[tuple[str, str]] = [

    # Darshan ticket types
    ("ದಿವ್ಯ ದರ್ಶನ",                      "Divya Darshan"),
    ("ಸರ್ವ ದರ್ಶನ",                       "Sarva Darshan"),
    ("ವಿಶೇಷ ದರ್ಶನ",                      "Vishesh Darshan"),
    ("ವಿಐಪಿ ದರ್ಶನ",                      "VIP Darshan"),

    # Seva types
    ("ಸುಪ್ರಭಾತಂ ಸೇವೆ",                   "Suprabhatam Seva"),
    ("ಸುಪ್ರಭಾತಂ",                         "Suprabhatam"),
    ("ಅರ್ಜಿತ ಸೇವೆ",                      "Arjita Seva"),
    ("ಕಲ್ಯಾಣೋತ್ಸವ",                     "Kalyanaotsavam"),

    # Prasadam
    ("ಲಡ್ಡು ಪ್ರಸಾದ",                     "Laddu Prasadam"),
    ("ಅನ್ನ ಪ್ರಸಾದ",                      "Anna Prasadam"),
    ("ಪ್ರಸಾದ",                            "Prasadam"),
    ("ಲಡ್ಡು",                              "Laddu"),

    # Festivals / events
    ("ಬ್ರಹ್ಮೋತ್ಸವ",                      "Brahmotsavam"),
    ("ವೈಕುಂಠ ಏಕಾದಶಿ",                   "Vaikunta Ekadasi"),

    # Places / org
    ("ತಿರುಮಲ ತಿರುಪತಿ ದೇವಸ್ಥಾನ",        "Tirumala Tirupati Devasthanams"),
    ("ತಿರುಮಲ",                            "Tirumala"),
    ("ತಿರುಪತಿ",                           "Tirupati"),
    ("ವೆಂಕಟೇಶ್ವರ",                        "Venkateswara"),
    ("ಬಾಲಾಜಿ",                            "Balaji"),

    # General terms
    ("ದರ್ಶನ",                              "Darshan"),
    ("ಸೇವೆ",                               "Seva"),
    ("ಹುಂಡಿ",                              "Hundi"),
    ("ಗೋಪುರ",                              "Gopuram"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Language → term list mapping
# ─────────────────────────────────────────────────────────────────────────────
_TERMS_BY_LANG: dict[str, list[tuple[str, str]]] = {
    "hi": _HI_TERMS,
    "te": _TE_TERMS,
    "ta": _TA_TERMS,
    "kn": _KN_TERMS,
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def mask_protected_terms(
    text: str,
    src_lang: str,
) -> tuple[str, dict[str, str]]:
    """
    Replace known domain proper-nouns with opaque placeholders.

    Args:
        text:     native-script query before translation
        src_lang: ISO 639-1 language code ("hi", "te", "ta", "kn")

    Returns:
        (masked_text, restore_map)
        masked_text  — text safe to pass to Google Translate
        restore_map  — {placeholder: canonical_english_name}
    """
    terms = _TERMS_BY_LANG.get(src_lang, [])
    if not terms:
        return text, {}

    restore_map: dict[str, str] = {}
    idx = 0

    for native, canonical in terms:
        if native in text:
            placeholder = f"{_PLACEHOLDER_PREFIX}{idx}{_PLACEHOLDER_SUFFIX}"
            text = text.replace(native, placeholder)
            restore_map[placeholder] = canonical
            idx += 1
            logger.debug("Masked [%s] %r → %r (restore: %r)", src_lang, native, placeholder, canonical)

    if restore_map:
        logger.info(
            "[ProtectedTerms] Shielded %d term(s) from translation: %s",
            len(restore_map),
            list(restore_map.values()),
        )

    return text, restore_map


def restore_protected_terms(
    translated: str,
    restore_map: dict[str, str],
) -> str:
    """
    Swap placeholder tokens back to their canonical English / transliterated names.

    Google Translate sometimes alters placeholder casing or surrounding punctuation,
    so we fall back to a case-insensitive regex match when a simple replace fails.

    Args:
        translated:  string returned by Google Translate (may contain placeholders)
        restore_map: {placeholder: canonical_english_name}

    Returns:
        Final translated string with all proper nouns correctly preserved.
    """
    if not restore_map:
        return translated

    for placeholder, canonical in restore_map.items():
        if placeholder in translated:
            translated = translated.replace(placeholder, canonical)
            logger.debug("Restored: %r → %r", placeholder, canonical)
        else:
            # Google Translate can lowercase, mangle underscores, etc.
            # Try a lenient case-insensitive search.
            pattern = re.compile(re.escape(placeholder), re.IGNORECASE)
            new = pattern.sub(canonical, translated)
            if new != translated:
                translated = new
                logger.debug("Restored (case-insensitive): %r → %r", placeholder, canonical)
            else:
                # Last resort: the placeholder was broken mid-word.
                # Log it and move on — better a missing restore than a crash.
                logger.warning(
                    "[ProtectedTerms] Could not restore placeholder %r after translation. "
                    "Google Translate may have fragmented it. Canonical: %r",
                    placeholder,
                    canonical,
                )

    return translated


def list_protected_terms(lang: str) -> list[tuple[str, str]]:
    """Return the full list of (native, canonical) pairs for a language (for inspection/testing)."""
    return list(_TERMS_BY_LANG.get(lang, []))
