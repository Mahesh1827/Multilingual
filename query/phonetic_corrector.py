"""
Comprehensive Phonetic Correction Layer for Indic ASR
=======================================================
Fixes Whisper speech-to-text phonetic errors in Telugu, Hindi, Tamil, Kannada.

Three-layer correction:
  1. Exact dictionary match    — instant, zero LLM cost
  2. Normalized fuzzy match    — catches vowel-length / aspirated-consonant variants
  3. LLM correction + translate — context-aware fix for everything else

Covers: numbers (1-100+), question words, grammar markers, Tirumala vocabulary,
        common verbs and nouns, administrative/member terms — in all 4 Indic scripts.
"""

import logging
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

_LANG_NAMES = {"te": "Telugu", "hi": "Hindi", "ta": "Tamil", "kn": "Kannada"}

# ═══════════════════════════════════════════════════════════════════════════════
# TELUGU  ( తెలుగు)
# ═══════════════════════════════════════════════════════════════════════════════
_TE: dict[str, str] = {

    # ── Numbers: 1-10 ─────────────────────────────────────────────────────────
    "ఒకట":          "ఒకటి",
    "ఒకటీ":         "ఒకటి",
    "ఒకటు":         "ఒకటి",
    "రేండు":        "రెండు",
    "రెండూ":        "రెండు",
    "రెందు":        "రెండు",
    "ముడు":         "మూడు",
    "మూడూ":         "మూడు",
    "మూడు":         "మూడు",          # correct – kept for normalization path
    "నాళుగు":       "నాలుగు",
    "నాలుగూ":       "నాలుగు",
    "అయిదూ":        "ఐదు",
    "అయిదు":        "ఐదు",
    "ఐదూ":          "ఐదు",
    "అరు":          "ఆరు",
    "ఆరూ":          "ఆరు",
    "ఎడు":          "ఏడు",
    "ఏడూ":          "ఏడు",
    "ఎనుమిది":      "ఎనిమిది",
    "ఎనమిది":       "ఎనిమిది",
    "ఎనమిదీ":       "ఎనిమిది",
    "ఎనిమిదీ":      "ఎనిమిది",
    "తమ్ముదు":      "తొమ్మిది",
    "తమ్మిది":      "తొమ్మిది",
    "తొమ్మిదు":     "తొమ్మిది",
    "తొమ్మిదీ":     "తొమ్మిది",
    "పదీ":          "పది",

    # ── Numbers: 11-19 ────────────────────────────────────────────────────────
    "పదకొండూ":      "పదకొండు",
    "పదకొండ":       "పదకొండు",
    "పన్నేండు":      "పన్నెండు",
    "పన్నెండూ":      "పన్నెండు",
    "పదముడు":       "పదమూడు",
    "పదమూడూ":       "పదమూడు",
    "పద్నాళుగు":    "పద్నాలుగు",
    "పదహేను":       "పదిహేను",
    "పదిహేనూ":      "పదిహేను",
    "పదహారూ":       "పదహారు",
    "పదహేరు":       "పదహారు",
    "పదహేడు":       "పదిహేడు",
    "పదిహేడూ":      "పదిహేడు",
    "పద్దెనిమిదీ":  "పద్దెనిమిది",
    "పద్దెనమిది":   "పద్దెనిమిది",
    "పంతొమ్మిదు":   "పంతొమ్మిది",
    "పంతొమ్మిదీ":   "పంతొమ్మిది",

    # ── Numbers: 20-100 ───────────────────────────────────────────────────────
    "ఇర్వే":         "ఇరవై",
    "ఇర్వయ్":        "ఇరవై",
    "ఇరవె":          "ఇరవై",
    "ఇరవయ్":         "ఇరవై",
    "ముప్పె":         "ముప్పై",
    "ముప్పెయ్":       "ముప్పై",
    "నలభె":           "నలభై",
    "నలభె":           "నలభై",
    "నలభె":           "నలభై",
    "ఏఁభై":           "యాభై",
    "యాభె":           "యాభై",
    "యాభెయ్":         "యాభై",
    "అరభై":           "అరవై",
    "అరబై":           "అరవై",
    "అరవె":           "అరవై",
    "డబ్భై":          "డెబ్బై",
    "డేబ్బై":         "డెబ్బై",
    "డెబ్బె":         "డెబ్బై",
    "ఎనభె":           "ఎనభై",
    "ఎనబై":           "ఎనభై",
    "తొంభె":          "తొంభై",
    "తొంభె":          "తొంభై",
    "వందా":           "వంద",
    "వందలు":          "వందలు",       # correct — kept for normalization

    # ── Question words ────────────────────────────────────────────────────────
    "ఎంటి":          "ఏమిటి",
    "ఎమిటి":         "ఏమిటి",
    "ఎమ్టి":         "ఏమిటి",
    "ఏమిటీ":         "ఏమిటి",
    "ఎన్ను":         "ఎన్ని",
    "ఎంను":          "ఎన్ని",
    "ఎన్నీ":         "ఎన్ని",
    "ఎంతో":          "ఎంత",
    "ఎక్కడో":        "ఎక్కడ",
    "ఎప్పుడో":       "ఎప్పుడు",
    "ఎందుకో":        "ఎందుకు",
    "ఏదో":           "ఏది",
    "ఏవో":           "ఏవి",
    "ఎవరో":          "ఎవరు",
    "ఎలాగో":         "ఎలా",
    "ఏమో":           "ఏమి",

    # ── Grammar markers / case endings ────────────────────────────────────────
    "అయింది":        "అయింది",       # correct
    "అయ్యింది":      "అయింది",
    "ఉన్నది":        "ఉంది",
    "ఉన్నది":        "ఉంది",
    "ఉన్నారూ":       "ఉన్నారు",
    "ఉన్నాయీ":       "ఉన్నాయి",
    "చేస్తారూ":      "చేస్తారు",
    "చేస్తుంది":     "చేస్తుంది",

    # ── Tirumala / TTD vocabulary ─────────────────────────────────────────────
    "తిరుపతీ":       "తిరుపతి",
    "తిరుమలా":       "తిరుమల",
    "వేంకటేస్వర":    "వేంకటేశ్వర",
    "వేంకటేస్వరా":   "వేంకటేశ్వర",
    "వేంకటేసుడు":    "వేంకటేశ్వరుడు",
    "సభ్భిలు":       "సభ్యులు",
    "సబ్బిలు":       "సభ్యులు",
    "సభ్యిలు":       "సభ్యులు",
    "సభ్యుళ్ళు":     "సభ్యులు",
    "కొందులు":       "కొండలు",
    "కొండులు":       "కొండలు",
    "పేరులు":        "పేర్లు",
    "పేర్లూ":        "పేర్లు",
    "దర్సనం":        "దర్శనం",
    "దర్సన":         "దర్శన",
    "దర్శనాలు":      "దర్శనాలు",    # correct
    "సేవలూ":         "సేవలు",
    "సేవా":          "సేవ",
    "ప్రసాదాం":      "ప్రసాదం",
    "ప్రసాదాలు":     "ప్రసాదాలు",
    "లద్దు":         "లడ్డూ",
    "లద్దూ":         "లడ్డూ",
    "లడ్డడు":        "లడ్డూ",
    "ఉష్కునోగ్రతే":  "ఉష్ణోగ్రత",
    "ఉష్ణోగ్రతే":    "ఉష్ణోగ్రత",
    "ఉష్ణోగ్రతా":    "ఉష్ణోగ్రత",
    "ఉష్నోగ్రత":     "ఉష్ణోగ్రత",
    "బ్రహ్మోత్సవాలూ": "బ్రహ్మోత్సవాలు",
    "రథోత్సవాలు":    "రథోత్సవాలు",
    "కళ్యానోత్సవం":  "కల్యాణోత్సవం",
    "కళ్యాణోత్సవాం": "కల్యాణోత్సవం",
    "వైకుంఠ":        "వైకుంఠ",
    "ఏకాదశి":        "ఏకాదశి",
    "ఆర్జిత":        "ఆర్జిత",
    "అర్చన":         "అర్చన",
    "అర్చనా":        "అర్చన",
    "హుండి":         "హుండీ",
    "గోపురాం":       "గోపురం",
    "విమానాం":       "విమానం",
    "మండపాం":        "మండపం",
    "ఆలయాం":        "ఆలయం",
    "దేవాలయాం":      "దేవాలయం",
    "ఆగమ":           "ఆగమ",
    "ఆగమశాస్త్రం":   "ఆగమశాస్త్రం",
    "శేషాచలాం":      "శేషాచలం",
    "వేదాచలాం":      "వేదాచలం",
    "గరుడాచలాం":     "గరుడాచలం",
    "అంజనాచలాం":     "అంజనాచలం",
    "నారాయణాచలాం":   "నారాయణాచలం",
    "వేంకటాచలాం":    "వేంకటాచలం",
    "వృషభాచలాం":     "వృషభాచలం",
    "సప్తగిరీ":      "సప్తగిరి",
    "సప్తగిరులూ":    "సప్తగిరులు",
    "అకాశగంగా":      "ఆకాశగంగ",
    "అకాసగంగ":       "ఆకాశగంగ",

    # ── Practical queries ─────────────────────────────────────────────────────
    "టైమింగులు":     "సమయాలు",
    "సమయాలూ":       "సమయాలు",
    "ధర":           "ధర",
    "ధరలు":         "ధరలు",
    "రుసుమూ":       "రుసుము",
    "రుసుమా":       "రుసుము",
    "బుకింగ్":       "బుకింగ్",
    "బుక్కింగ్":     "బుకింగ్",
    "టికెట్టు":      "టికెట్",
    "బస్సు":        "బస్సు",
    "హెలీకాప్టరు":  "హెలికాప్టర్",
    "హెలీకాప్టర":   "హెలికాప్టర్",
    "దారి":         "దారి",
    "దూరాం":        "దూరం",
    "చేరుకోవాలంటే": "చేరుకోవాలంటే",
    "ఉండటానికి":    "ఉండటానికి",
    "కాటేజీ":       "కాటేజ్",
    "లాడ్జి":       "లాడ్జ్",
    "భోజనాం":       "భోజనం",
    "అన్నప్రసాదాం":  "అన్నప్రసాదం",

    # ── Common adjectives / modifiers ────────────────────────────────────────
    "అన్ని":        "అన్ని",
    "అన్నీ":        "అన్ని",
    "అందరూ":        "అందరు",
    "చాలా":         "చాలా",
    "కొన్నీ":       "కొన్ని",
    "మొత్తాం":      "మొత్తం",
    "ప్రస్తుతాం":   "ప్రస్తుతం",
    "ప్రస్తుత":     "ప్రస్తుత",

    # ── Common verb forms ─────────────────────────────────────────────────────
    "తెలుసుకోవాలి": "తెలుసుకోవాలి",
    "చెప్పండీ":     "చెప్పండి",
    "చెప్పు":       "చెప్పు",
    "చెప్పాడు":     "చెప్పాడు",
    "అడిగాడూ":      "అడిగాడు",
    "వస్తున్నారూ":  "వస్తున్నారు",
    "వెళ్ళాలి":    "వెళ్ళాలి",
    "వెళ్ళండి":    "వెళ్ళండి",
    "చూడాలి":      "చూడాలి",
    "తెలుసా":      "తెలుసా",
    "తెలుసుకో":    "తెలుసుకో",
}

# ═══════════════════════════════════════════════════════════════════════════════
# HINDI  (हिंदी)
# ═══════════════════════════════════════════════════════════════════════════════
_HI: dict[str, str] = {

    # ── Numbers: 1-10 ─────────────────────────────────────────────────────────
    "एक":       "एक",
    "दो":       "दो",
    "तीन":      "तीन",
    "चार":      "चार",
    "पाँच":     "पाँच",
    "पांच":     "पाँच",
    "छः":       "छह",
    "छे":       "छह",
    "सात":      "सात",
    "आठ":       "आठ",
    "नौ":       "नौ",
    "दस":       "दस",

    # ── Numbers: 11-20 ────────────────────────────────────────────────────────
    "ग्यारह":   "ग्यारह",
    "गियारह":   "ग्यारह",
    "बारह":     "बारह",
    "तेरह":     "तेरह",
    "चौदह":     "चौदह",
    "पंद्रह":   "पंद्रह",
    "पन्द्रह":  "पंद्रह",
    "सोलह":     "सोलह",
    "सत्रह":    "सत्रह",
    "अठारह":    "अठारह",
    "उन्नीस":   "उन्नीस",
    "बीस":      "बीस",

    # ── Tens ──────────────────────────────────────────────────────────────────
    "तीस":      "तीस",
    "चालीस":    "चालीस",
    "पचास":     "पचास",
    "साठ":      "साठ",
    "सत्तर":    "सत्तर",
    "अस्सी":    "अस्सी",
    "नब्बे":    "नब्बे",
    "सौ":       "सौ",

    # ── Common compound numbers (ASR error variants) ──────────────────────────
    "उन्तिस":   "उनतीस",    # 29
    "उन्तीस":   "उनतीस",    # 29
    "उनतिस":    "उनतीस",    # 29
    "अट्ठाइस":  "अट्ठाईस",  # 28
    "सत्तइस":   "सत्ताईस",  # 27
    "बत्तिस":   "बत्तीस",   # 32
    "सैंतिस":   "सैंतीस",   # 37
    "उनचास":    "उनचास",    # 49

    # ── Question words ────────────────────────────────────────────────────────
    "क्या":     "क्या",
    "कितनी":    "कितने",
    "कितना":    "कितना",
    "कहाँ":     "कहाँ",
    "कहा":      "कहाँ",
    "कब":       "कब",
    "क्यों":    "क्यों",
    "क्यो":     "क्यों",
    "कैसे":     "कैसे",
    "कैसा":     "कैसा",
    "कोनसे":    "कौनसे",
    "कोनसा":    "कौनसा",
    "कोन":      "कौन",
    "कौनसी":    "कौनसी",

    # ── Tirumala vocabulary ───────────────────────────────────────────────────
    "तिरुपती":   "तिरुपति",
    "तिरुमला":   "तिरुमला",
    "वेंकटेश्वर": "वेंकटेश्वर",
    "बालाजी":    "बालाजी",
    "सदस्य":     "सदस्य",
    "मेम्बर":    "सदस्य",
    "पर्बत":     "पहाड़",
    "पहाड़":     "पहाड़",
    "सात":       "सात",
    "पर्वत":     "पहाड़",
    "दर्शन":     "दर्शन",
    "दर्सन":     "दर्शन",
    "सेवा":      "सेवा",
    "प्रसाद":    "प्रसाद",
    "लड्डू":     "लड्डू",
    "लडू":       "लड्डू",
    "तापमान":    "तापमान",
    "ब्रह्मोत्सव": "ब्रह्मोत्सव",
    "ब्रम्होत्सव": "ब्रह्मोत्सव",
    "मंदिर":     "मंदिर",
    "मन्दिर":    "मंदिर",
    "बुकिंग":    "बुकिंग",
    "टिकट":      "टिकट",
    "नाम":       "नाम",
    "नाम":       "नाम",

    # ── Common words ──────────────────────────────────────────────────────────
    "वर्तमान":   "वर्तमान",
    "अभि":       "अभी",
    "मौसम":      "मौसम",
    "जानकारी":   "जानकारी",
    "बताइए":     "बताइए",
    "बताओ":      "बताओ",
    "समय":       "समय",
    "टाइमिंग":   "समय",
    "कीमत":      "कीमत",
    "शुल्क":     "शुल्क",
    "कैसे पहुँचें": "कैसे पहुँचें",
    "इतिहास":    "इतिहास",
    "कुल":       "कुल",
    "सभी":       "सभी",
}

# ═══════════════════════════════════════════════════════════════════════════════
# TAMIL  (தமிழ்)
# ═══════════════════════════════════════════════════════════════════════════════
_TA: dict[str, str] = {

    # ── Numbers: 1-10 ─────────────────────────────────────────────────────────
    "ஒன்று":         "ஒன்று",
    "இரண்டு":        "இரண்டு",
    "மூன்று":        "மூன்று",
    "நான்கு":        "நான்கு",
    "ஐந்து":         "ஐந்து",
    "ஆறு":           "ஆறு",
    "ஏழு":           "ஏழு",
    "எட்டு":         "எட்டு",
    "ஒன்பது":        "ஒன்பது",
    "ஒம்பது":        "ஒன்பது",        # ASR variant
    "பத்து":         "பத்து",

    # ── Tens ──────────────────────────────────────────────────────────────────
    "இருபது":        "இருபது",
    "இருபத்தி":      "இருபது",
    "முப்பது":       "முப்பது",
    "நாற்பது":       "நாற்பது",
    "ஐம்பது":        "ஐம்பது",
    "அறுபது":        "அறுபது",
    "எழுபது":        "எழுபது",
    "எண்பது":        "எண்பது",
    "தொண்ணூறு":      "தொண்ணூறு",
    "நூறு":          "நூறு",

    # ── Common ASR variants for compound numbers ──────────────────────────────
    "இருபத்துஒன்பது": "இருபத்தொன்பது",   # 29

    # ── Question words ────────────────────────────────────────────────────────
    "என்ன":          "என்ன",
    "என்னா":         "என்ன",
    "எவ்வளவு":       "எவ்வளவு",
    "எவ்வளவா":       "எவ்வளவு",
    "எங்கே":         "எங்கே",
    "எங்கு":         "எங்கு",
    "எப்போது":       "எப்போது",
    "எப்பொழுது":     "எப்போது",
    "ஏன்":           "ஏன்",
    "எப்படி":        "எப்படி",
    "யார்":          "யார்",
    "எது":           "எது",
    "எதுவும்":       "எதுவும்",
    "எத்தனை":        "எத்தனை",

    # ── Tirumala vocabulary ───────────────────────────────────────────────────
    "திருமலை":       "திருமலை",
    "திருப்பதி":     "திருப்பதி",
    "வேங்கடேஸ்வர":  "வேங்கடேஸ்வர",
    "வேங்கடேஸ்வரர்": "வேங்கடேஸ்வரர்",
    "கோவில்":        "கோயில்",
    "கோவிலு":        "கோயில்",
    "தேவஸ்தானம்":   "தேவஸ்தானம்",
    "தரிசனம்":       "தரிசனம்",
    "தரிசன":         "தரிசன",
    "சேவை":          "சேவை",
    "பிரசாதம்":      "பிரசாதம்",
    "லட்டு":         "லட்டு",
    "உறுப்பினர்":    "உறுப்பினர்கள்",
    "உறுப்பினர்க":   "உறுப்பினர்கள்",
    "பெயர்கள்":      "பெயர்கள்",
    "பேர்கள":        "பெயர்கள்",
    "மலைகள்":        "மலைகள்",
    "மலைக":          "மலைகள்",
    "வெப்பநிலை":     "வெப்பநிலை",
    "வெப்பம்":       "வெப்பநிலை",
    "வானிலை":        "வானிலை",
    "பிரம்மோத்சவம்": "பிரம்மோத்சவம்",
    "பிரம்மோத்சவாம்": "பிரம்மோத்சவம்",
    "வைகுண்ட":       "வைகுண்ட",
    "ஏகாதசி":        "ஏகாதசி",
    "ஏகாதசீ":        "ஏகாதசி",
    "நேரம்":         "நேரம்",
    "நேரங்கள்":      "நேரங்கள்",
    "கட்டணம்":       "கட்டணம்",
    "விலை":          "விலை",
    "முன்பதிவு":     "முன்பதிவு",
    "வரலாறு":        "வரலாறு",
    "மொத்தம்":       "மொத்தம்",
    "தற்போது":       "தற்போது",
    "இப்போது":       "இப்போது",
    "இப்போதய":       "இப்போது",
}

# ═══════════════════════════════════════════════════════════════════════════════
# KANNADA  (ಕನ್ನಡ)
# ═══════════════════════════════════════════════════════════════════════════════
_KN: dict[str, str] = {

    # ── Numbers: 1-10 ─────────────────────────────────────────────────────────
    "ಒಂದು":           "ಒಂದು",
    "ಒಂದೂ":           "ಒಂದು",
    "ಎರಡು":           "ಎರಡು",
    "ಎರಡೂ":           "ಎರಡು",
    "ಮೂರು":           "ಮೂರು",
    "ಮೂರೂ":           "ಮೂರು",
    "ನಾಲ್ಕು":         "ನಾಲ್ಕು",
    "ನಾಲ್ಕೂ":         "ನಾಲ್ಕು",
    "ಐದು":            "ಐದು",
    "ಆರು":            "ಆರು",
    "ಏಳು":            "ಏಳು",
    "ಎಳು":            "ಏಳು",
    "ಎಂಟು":           "ಎಂಟು",
    "ಒಂಬತ್ತು":        "ಒಂಬತ್ತು",
    "ಒಂಬತ್ತೂ":        "ಒಂಬತ್ತು",
    "ಹತ್ತು":          "ಹತ್ತು",

    # ── Tens ──────────────────────────────────────────────────────────────────
    "ಇಪ್ಪತ್ತು":        "ಇಪ್ಪತ್ತು",
    "ಮೂವತ್ತು":         "ಮೂವತ್ತು",
    "ನಲವತ್ತು":         "ನಲವತ್ತು",
    "ಐವತ್ತು":          "ಐವತ್ತು",
    "ಅರವತ್ತು":         "ಅರವತ್ತು",
    "ಎಪ್ಪತ್ತು":        "ಎಪ್ಪತ್ತು",
    "ಎಂಬತ್ತು":         "ಎಂಬತ್ತು",
    "ತೊಂಬತ್ತು":        "ತೊಂಬತ್ತು",
    "ನೂರು":            "ನೂರು",

    # ── Question words ────────────────────────────────────────────────────────
    "ಏನು":            "ಏನು",
    "ಎಷ್ಟು":          "ಎಷ್ಟು",
    "ಎಂಟ":            "ಎಷ್ಟು",
    "ಎಲ್ಲಿ":          "ಎಲ್ಲಿ",
    "ಯಾವಾಗ":          "ಯಾವಾಗ",
    "ಏಕೆ":            "ಏಕೆ",
    "ಹೇಗೆ":           "ಹೇಗೆ",
    "ಯಾರು":           "ಯಾರು",
    "ಯಾವುದು":         "ಯಾವುದು",
    "ಎಷ್ಟು":          "ಎಷ್ಟು",
    "ಎಂತಹ":           "ಎಂತಹ",
    "ಎಷ್ಟೋ":          "ಎಷ್ಟು",

    # ── Tirumala vocabulary ───────────────────────────────────────────────────
    "ತಿರುಮಲ":         "ತಿರುಮಲ",
    "ತಿರುಪತಿ":        "ತಿರುಪತಿ",
    "ವೆಂಕಟೇಶ್ವರ":     "ವೆಂಕಟೇಶ್ವರ",
    "ವೆಂಕಟೇಸ್ವರ":     "ವೆಂಕಟೇಶ್ವರ",
    "ದೇವಸ್ಥಾನ":       "ದೇವಸ್ಥಾನ",
    "ದೇವಸ್ತಾನ":       "ದೇವಸ್ಥಾನ",
    "ದೇವಾಲಯ":         "ದೇವಾಲಯ",
    "ದರ್ಶನ":           "ದರ್ಶನ",
    "ದರ್ಶಣ":           "ದರ್ಶನ",
    "ಸೇವೆ":            "ಸೇವೆ",
    "ಪ್ರಸಾದ":          "ಪ್ರಸಾದ",
    "ಪ್ರಸಾದಾ":         "ಪ್ರಸಾದ",
    "ಲಡ್ಡು":           "ಲಡ್ಡು",
    "ಲಡ್ಡೂ":           "ಲಡ್ಡು",
    "ಸದಸ್ಯರು":         "ಸದಸ್ಯರು",
    "ಸದಸ್ಯರ":          "ಸದಸ್ಯರ",
    "ಹೆಸರುಗಳ":         "ಹೆಸರುಗಳು",
    "ಹೆಸರುಗಳ":         "ಹೆಸರುಗಳು",
    "ಬೆಟ್ಟಗಳ":         "ಬೆಟ್ಟಗಳು",
    "ತಾಪಮಾನ":          "ತಾಪಮಾನ",
    "ಉಷ್ಣತೆ":          "ತಾಪಮಾನ",
    "ಹವಾಮಾನ":          "ಹವಾಮಾನ",
    "ಬ್ರಹ್ಮೋತ್ಸವ":      "ಬ್ರಹ್ಮೋತ್ಸವ",
    "ಬ್ರಮ್ಹೋತ್ಸವ":      "ಬ್ರಹ್ಮೋತ್ಸವ",
    "ಸಮಯ":             "ಸಮಯ",
    "ಸಮಯಗಳು":          "ಸಮಯಗಳು",
    "ಶುಲ್ಕ":           "ಶುಲ್ಕ",
    "ಬೆಲೆ":            "ಬೆಲೆ",
    "ಕಾಯ್ದಿರಿಸುವಿಕೆ":  "ಬುಕಿಂಗ್",
    "ಇತಿಹಾಸ":          "ಇತಿಹಾಸ",
    "ಒಟ್ಟು":           "ಒಟ್ಟು",
    "ಪ್ರಸ್ತುತ":         "ಪ್ರಸ್ತುತ",
    "ಈಗ":              "ಈಗ",
}

_PHONETIC_DICTS: dict[str, dict] = {
    "te": _TE, "hi": _HI, "ta": _TA, "kn": _KN
}

# ═══════════════════════════════════════════════════════════════════════════════
# Language-specific phonetic normalization
# Maps characters that are acoustically identical/similar to a canonical form.
# Lets "ఇరవె" match "ఇరవై" even if not in the dictionary.
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_te(w: str) -> str:
    """Reduce Telugu word to phonetically canonical form for fuzzy matching."""
    t = w
    # Long vowels → short
    for long, short in [("ఆ","అ"),("ఈ","ఇ"),("ఊ","ఉ"),("ఏ","ఎ"),("ఓ","ఒ"),("ఐ","అఇ"),("ఔ","అఉ")]:
        t = t.replace(long, short)
    # Aspirated → unaspirated
    for asp, unasp in [("భ","బ"),("ఘ","గ"),("ధ","ద"),("ఝ","జ"),("థ","త"),("ఖ","క"),("ఫ","ప"),("ఛ","చ"),("ఢ","డ")]:
        t = t.replace(asp, unasp)
    # Retroflex → dental (approximate)
    for ret, den in [("ణ","న"),("ళ","ల"),("ష","స"),("ఠ","త")]:
        t = t.replace(ret, den)
    return t

def _normalize_hi(w: str) -> str:
    """Reduce Hindi word to phonetically canonical form."""
    t = w
    # Nasal marks unification
    t = t.replace("ँ", "ं")
    # Schwa unification (drop final ा where it might be implicit)
    # Long→short vowels
    for long, short in [("आ","अ"),("ई","इ"),("ऊ","उ"),("ए","ऐ"),("ओ","औ")]:
        t = t.replace(long, short)
    # Aspirated→unaspirated
    for asp, unasp in [("ख","क"),("घ","ग"),("छ","च"),("झ","ज"),("ठ","ट"),("ढ","ड"),("थ","त"),("ध","द"),("फ","प"),("भ","ब")]:
        t = t.replace(asp, unasp)
    return t

def _normalize_ta(w: str) -> str:
    """Reduce Tamil word to phonetically canonical form."""
    t = w
    # Long→short vowels
    for long, short in [("ஆ","அ"),("ஈ","இ"),("ஊ","உ"),("ஏ","எ"),("ஓ","ஒ"),("ஐ","அஇ"),("ஔ","அஉ")]:
        t = t.replace(long, short)
    return t

def _normalize_kn(w: str) -> str:
    """Reduce Kannada word to phonetically canonical form."""
    t = w
    # Long→short vowels
    for long, short in [("ಆ","ಅ"),("ಈ","ಇ"),("ಊ","ಉ"),("ಏ","ಎ"),("ಓ","ಒ"),("ಐ","ಅಇ"),("ಔ","ಅಉ")]:
        t = t.replace(long, short)
    # Aspirated→unaspirated
    for asp, unasp in [("ಭ","ಬ"),("ಘ","ಗ"),("ಧ","ದ"),("ಝ","ಜ"),("ಥ","ತ"),("ಖ","ಕ"),("ಫ","ಪ"),("ಛ","ಚ"),("ಢ","ಡ")]:
        t = t.replace(asp, unasp)
    return t

_NORMALIZERS = {
    "te": _normalize_te,
    "hi": _normalize_hi,
    "ta": _normalize_ta,
    "kn": _normalize_kn,
}

# ═══════════════════════════════════════════════════════════════════════════════
# LLM correction prompt — comprehensive phonetic guidance per language
# ═══════════════════════════════════════════════════════════════════════════════

_LANG_PHONETIC_GUIDES: dict[str, str] = {
    "te": """\
Telugu phonetic correction rules:
- Vowel length: Short (అ,ఇ,ఉ,ఎ,ఒ) and long (ఆ,ఈ,ఊ,ఏ,ఓ) are often swapped by Whisper.
- Aspirated ↔ unaspirated: భ/బ, ఘ/గ, ధ/ద, థ/త, ఖ/క, ఫ/ప, ఛ/చ, ఝ/జ
- Retroflex ↔ dental: ణ/న, ళ/ల, ట/త, డ/ద
- Numbers: ఇర్వే→ఇరవై(20), తమ్ముదు→తొమ్మిది(9), ముప్పె→ముప్పై(30), నలభె→నలభై(40)
- Question words: ఎంటి→ఏమిటి, ఎన్ను→ఎన్ని
- Members: సబ్బిలు/సభ్భిలు→సభ్యులు | Temperature: ఉష్కునోగ్రతే→ఉష్ణోగ్రత
- Fix ALL grammatical case endings (-లు/-ల/-కి/-కు/-లో/-పై) to match the sentence context.""",

    "hi": """\
Hindi phonetic correction rules:
- Nasalization: ं/ँ often missing or added incorrectly.
- Aspirated ↔ unaspirated: ख/क, घ/ग, छ/च, झ/ज, थ/त, ध/द, फ/प, भ/ब
- Schwa deletion: final 'a' (अ) is often silent — Whisper may keep or drop it incorrectly.
- Numbers: compound numbers like उनतीस(29), अट्ठाईस(28) are often misheard.
- Question words: कितनी→कितने, कोनसे→कौनसे, कोन→कौन
- Fix postpositions (में, को, का, की, के, पर, से) to match grammatical context.""",

    "ta": """\
Tamil phonetic correction rules:
- Vowel length: Short (அ,இ,உ,எ,ஒ) vs long (ஆ,ஈ,ஊ,ஏ,ஓ) are often swapped.
- Hard/soft consonant pairs: க/ககூ — Tamil has context-dependent pronunciation.
- Compound numbers: இருபத்தொன்பது(29) often misheard.
- Question words: என்ன, எவ்வளவு, எங்கே, எப்போது, ஏன், எப்படி
- Members: உறுப்பினர்கள் | Names: பெயர்கள் (not பேர்கள)
- Temperature: வெப்பநிலை (not வெப்பம்)
- Fix case suffixes (-ல்,-ஐ,-கு,-இன்,-உடன்) for grammatical correctness.""",

    "kn": """\
Kannada phonetic correction rules:
- Vowel length: Short (ಅ,ಇ,ಉ,ಎ,ಒ) vs long (ಆ,ಈ,ಊ,ಏ,ಓ) often confused.
- Aspirated ↔ unaspirated: ಭ/ಬ, ಘ/ಗ, ಧ/ದ, ಥ/ತ, ಖ/ಕ, ಫ/ಪ
- Question words: ಏನು, ಎಷ್ಟು (not ಎಂಟ), ಎಲ್ಲಿ, ಯಾವಾಗ, ಏಕೆ, ಹೇಗೆ, ಯಾರು
- Temple: ದೇವಸ್ತಾನ→ದೇವಸ್ಥಾನ, ದರ್ಶಣ→ದರ್ಶನ
- Members: ಸದಸ್ಯರು | Temperature: ಉಷ್ಣತೆ→ತಾಪಮಾನ
- Fix case suffixes (-ಅನ್ನು,-ಗೆ,-ಇಂದ,-ಲ್ಲಿ,-ಉ) for grammatical correctness.""",
}

_CORRECTION_PROMPT = """\
You are an expert in {lang_name} speech-to-text error correction.

The input is a Whisper ASR transcription that may contain phonetic spelling mistakes.
Context: question about Tirumala Temple (darshan, seva, members, hills, temperature, history, etc.)

Phonetic correction guide for {lang_name}:
{phonetic_guide}

TASK: Correct EVERY phonetically wrong word in the sentence.
Do not guess meanings — only fix clear phonetic/spelling errors.
Produce naturally spoken, grammatically correct {lang_name}.

Output EXACTLY two lines (no explanations, no extra text):
Corrected: <corrected {lang_name} sentence>
English: <English translation>

Input: {text}"""


# ═══════════════════════════════════════════════════════════════════════════════
# Core correction functions
# ═══════════════════════════════════════════════════════════════════════════════

def _exact_dict_apply(text: str, lang_dict: dict[str, str]) -> tuple[str, int]:
    """Token-level exact dictionary replacement."""
    words  = text.split()
    fixed  = [lang_dict.get(w, w) for w in words]
    n      = sum(1 for o, f in zip(words, fixed) if o != f)
    return " ".join(fixed), n


def _normalized_dict_apply(text: str, lang: str, lang_dict: dict[str, str]) -> tuple[str, int]:
    """
    Normalized fuzzy match: normalize both input word and all dictionary keys
    to a canonical phonetic form, then do exact lookup on normalized form.
    Catches vowel-length and aspirated-consonant variants not in exact dict.
    """
    normalize = _NORMALIZERS.get(lang)
    if not normalize:
        return text, 0

    # Build normalized → original mapping for the dictionary values
    norm_key_to_value: dict[str, str] = {}
    for wrong, correct in lang_dict.items():
        norm_key_to_value[normalize(wrong)] = correct

    words = text.split()
    fixed = []
    n = 0
    for w in words:
        norm_w = normalize(w)
        if norm_w in norm_key_to_value and norm_key_to_value[norm_w] != w:
            fixed.append(norm_key_to_value[norm_w])
            n += 1
        else:
            fixed.append(w)
    return " ".join(fixed), n


def apply_phonetic_dict(text: str, lang: str) -> str:
    """
    Apply language-specific phonetic dictionary.
    Two-pass: exact match first, then normalized fuzzy match.
    """
    lang_dict = _PHONETIC_DICTS.get(lang, {})
    if not lang_dict:
        return text

    text, n1 = _exact_dict_apply(text, lang_dict)
    text, n2 = _normalized_dict_apply(text, lang, lang_dict)
    total = n1 + n2
    if total:
        logger.info("Phonetic dict [%s]: %d correction(s) (exact=%d, fuzzy=%d)", lang, total, n1, n2)
    return text


def llm_correct_and_translate(text: str, lang: str) -> dict | None:
    """
    LLM-based phonetic correction + English translation.
    Returns {"corrected": str, "english": str} or None on failure.
    """
    lang_name     = _LANG_NAMES.get(lang, lang.upper())
    phonetic_guide = _LANG_PHONETIC_GUIDES.get(lang, "Fix phonetic and spelling errors.")

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from query.agents.knowledge_rag_agent import _get_llm

        prompt = ChatPromptTemplate.from_messages([("human", _CORRECTION_PROMPT)])
        chain  = prompt | _get_llm() | StrOutputParser()
        raw    = chain.invoke({
            "lang_name":      lang_name,
            "phonetic_guide": phonetic_guide,
            "text":           text,
        }).strip()

        corrected = english = None
        for line in raw.splitlines():
            if line.startswith("Corrected:"):
                corrected = line[len("Corrected:"):].strip()
            elif line.startswith("English:"):
                english = line[len("English:"):].strip()

        if corrected and english:
            return {"corrected": corrected, "english": english}
        logger.warning("LLM correction: unexpected format — %r", raw[:120])

    except Exception as e:
        logger.warning("LLM phonetic correction failed: %s", e)

    return None


def correct_and_translate(text: str, lang: str) -> dict:
    """
    Main entry point.

    Layer 1: Exact dictionary  — zero cost, handles all known ASR confusions
    Layer 2: Normalized fuzzy  — catches vowel-length / aspirated variants
    Layer 3: LLM correction    — catches everything else + provides English translation

    Returns:
        {"corrected": str,   # corrected native text for display
         "english":   str}   # English translation for RAG (empty if LLM unavailable)
    """
    if not text or lang == "en":
        return {"corrected": text, "english": ""}

    # Layers 1+2: dictionary (exact + normalized fuzzy)
    dict_corrected = apply_phonetic_dict(text, lang)

    # Layer 3: LLM — corrects remaining errors AND translates
    llm_result = llm_correct_and_translate(dict_corrected, lang)

    if llm_result:
        final = llm_result["corrected"] or dict_corrected
        if final != text:
            logger.info("Phonetic correction [%s]:\n  IN : %s\n  OUT: %s", lang, text, final)
        return {"corrected": final, "english": llm_result["english"]}

    # LLM unavailable — return dictionary-corrected text; English via Whisper Pass 3
    return {"corrected": dict_corrected, "english": ""}
