"""
stt_module.py
=============
Real-time Speech-to-Text for the Multilingual Query Architecture.

Features:
  - Microphone capture via Push-to-Talk (manual block reading)
  - Transcription via faster-whisper (local, no API key needed)
  - Language detection via langdetect (post-transcription fallback)
  - Whisper natively detects language during transcription (primary method)
  - Returns: {"text": str, "detected_lang": str (ISO 639-1)}

Install dependencies:
    pip install faster-whisper sounddevice numpy lingua-language-detector
"""

import re
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Domain-specific Whisper initial prompts
#
# Providing these as `initial_prompt` on the second (refinement) pass
# biases Whisper's decoder toward Tirumala vocabulary, fixing errors like:
#   "ఎన్ను" → "ఎన్ని"   (how many)
#   "ఉష్కునోగ్రతే" → "ఉష్ణోగ్రత"  (temperature)
# ─────────────────────────────────────────────
_DOMAIN_PROMPT: dict[str, str] = {
    "te": (
        "తిరుమల తిరుపతి వేంకటేశ్వర దేవాలయం "
        "ప్రస్తుత ఉష్ణోగ్రత ఎంత ఇప్పుడు ఎప్పుడు ఎన్ని ఎక్కడ ఎందుకు "
        "కొండలు ఏడు పేర్లు శేషాచలం వేదాచలం గరుడాచలం "
        "అంజనాచలం వృషభాచలం నారాయణాచలం వేంకటాచలం "
        "దర్శనం సేవ ప్రసాదం లడ్డూ బ్రహ్మోత్సవం"
    ),
    "hi": (
        "तिरुमला तिरुपति वेंकटेश्वर मंदिर "
        "वर्तमान तापमान अभी अभी कितना कब कहाँ क्यों कितने "
        "पहाड़ सात नाम दर्शन सेवा प्रसाद लड्डू ब्रह्मोत्सव"
    ),
    "ta": (
        "திருமலை திருப்பதி வேங்கடேஸ்வர கோயில் "
        "தற்போது வெப்பநிலை இப்போது எவ்வளவு எப்போது எங்கே ஏன் எத்தனை "
        "மலைகள் ஏழு பெயர்கள் தரிசனம் சேவை பிரசாதம் பிரம்மோத்சவம்"
    ),
    "kn": (
        "ತಿರುಮಲ ತಿರುಪತಿ ವೆಂಕಟೇಶ್ವರ ದೇವಾಲಯ "
        "ಪ್ರಸ್ತುತ ತಾಪಮಾನ ಈಗ ಎಷ್ಟು ಯಾವಾಗ ಎಲ್ಲಿ ಏಕೆ ಎಷ್ಟು "
        "ಬೆಟ್ಟಗಳು ಏಳು ಹೆಸರುಗಳು ದರ್ಶನ ಸೇವೆ ಪ್ರಸಾದ ಬ್ರಹ್ಮೋತ್ಸವ"
    ),
    "en": (
        "Tirumala Tirupati Venkateswara temple "
        "current temperature now today how much how many when where why "
        "hills seven names Seshachalam Vedachalam Garudachalam Anjanachalam "
        "Vrishabhachalam Narayanachalam Venkatachalam darshan seva prasadam laddu"
    ),
}

_ENGLISH_DOMAIN_PROMPT = _DOMAIN_PROMPT["en"]

# ─────────────────────────────────────────────
# Lingua language detector (lazy-loaded)
# ─────────────────────────────────────────────
_lingua_detector = None

# Map from lingua Language enum name → ISO 639-1 code
_LINGUA_TO_ISO = {
    "ENGLISH": "en", "HINDI": "hi", "TELUGU": "te",
    "TAMIL": "ta", "KANNADA": "kn"
}

def _get_lingua_detector():
    """Lazy-load lingua detector. Dynamically checks which languages exist."""
    global _lingua_detector
    if _lingua_detector is None:
        try:
            from lingua import Language, LanguageDetectorBuilder
            # Dynamically resolve only languages that exist in this version
            supported = []
            for lang_name in _LINGUA_TO_ISO:
                if hasattr(Language, lang_name):
                    supported.append(getattr(Language, lang_name))
            if not supported:
                logger.warning("Lingua: No supported languages found.")
                return None
            _lingua_detector = LanguageDetectorBuilder.from_languages(
                *supported
            ).with_minimum_relative_distance(0.10).build()
            logger.info(f"Lingua language detector loaded ({len(supported)} languages).")
        except Exception as e:
            logger.warning(f"Lingua init failed: {e}. Falling back to Whisper.")
    return _lingua_detector


def _detect_language_lingua(text: str, min_relative_distance: float = 0.10) -> str | None:
    """
    Detect language using lingua-py. Returns ISO 639-1 code or None.

    Args:
        min_relative_distance: Confidence margin between top-2 candidates.
            Lower = more sensitive (better for short 1–3 word queries).
            Default 0.10 (was 0.15 which was too strict).
    """
    detector = _get_lingua_detector()
    if detector is None:
        return None
    try:
        result = detector.detect_language_of(text)
        if result is not None:
            return _LINGUA_TO_ISO.get(result.name, None)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# Unicode script-range detector (fastest + most reliable for native text)
# ─────────────────────────────────────────────

# (lo, hi, iso_code)
_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0900, 0x097F, "hi"),   # Devanagari → Hindi
    (0x0C00, 0x0C7F, "te"),   # Telugu
    (0x0B80, 0x0BFF, "ta"),   # Tamil
    (0x0C80, 0x0CFF, "kn"),   # Kannada
]


def _detect_script_language(text: str) -> str | None:
    """
    Detect language purely from Unicode script block ranges.

    This is the fastest and most reliable method for native-script text.
    Completely immune to phonetic / acoustic confusion.

    Returns ISO 639-1 code or None (Latin/ASCII input returns None).
    """
    if not text:
        return None

    counts: dict[str, int] = {}
    total_non_ascii = 0

    for ch in text:
        cp = ord(ch)
        if cp > 127:
            total_non_ascii += 1
        for lo, hi, lang in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break

    if not counts:
        return None  # All ASCII — not native-script

    best_lang = max(counts, key=counts.__getitem__)
    best_count = counts[best_lang]

    # Require ≥2 script chars OR ≥20% of all non-ASCII chars
    threshold = max(2, int(total_non_ascii * 0.20))
    if best_count >= threshold:
        return best_lang

    return None


# ─────────────────────────────────────────────
# LANGUAGE CORRECTION / REMAPPING RULES
# (Tirumala Assistant — Rule 2 & 3)
# ─────────────────────────────────────────────

# Punjabi indicator words commonly confused with Kannada/Hindi
_KANNADA_INDICATOR_WORDS = {
    "ಏನು", "ಇದು", "ಅದು", "ನಾನು", "ನೀವು", "ಅವರು", "ಎಷ್ಟು",
    "ದೇವಾಲಯ", "ದರ್ಶನ", "ತಿರುಮಲ", "ತಿರುಪತಿ", "ಸೇವೆ",
}
_HINDI_INDICATOR_WORDS = {
    "क्या", "है", "हैं", "यह", "वह", "हम", "आप", "मैं", "कितना",
    "दर्शन", "मंदिर", "तिरुमला", "तिरुपति", "सेवा",
}


def _correct_language(
    whisper_lang: str,
    confidence: float,
    text: str,
) -> tuple[str, str]:
    """
    Apply Tirumala assistant language correction rules.

    Returns:
        (corrected_lang, reason) — reason is a short log string.
    """
    SUPPORTED = {"en", "hi", "te", "ta", "kn"}

    # Rule 2a: Urdu → Hindi (spoken forms are nearly identical)
    if whisper_lang == "ur":
        return "hi", "ur→hi (Urdu remapped to Hindi)"

    # Rule 2b: Punjabi with low confidence → try Lingua, then heuristic
    if whisper_lang == "pa" and confidence < 0.75:
        lingua_lang = _detect_language_lingua(text)
        if lingua_lang and lingua_lang in SUPPORTED:
            return lingua_lang, f"pa+low-conf → Lingua={lingua_lang}"
        # Heuristic word-based disambiguation
        words = set(text.split())
        if words & _KANNADA_INDICATOR_WORDS:
            return "kn", "pa+low-conf → heuristic=kn (Kannada indicators found)"
        return "hi", "pa+low-conf → heuristic=hi (default)"

    # Rule 2c: Low confidence on any unsupported language → Lingua re-evaluation
    if whisper_lang not in SUPPORTED and confidence < 0.70:
        lingua_lang = _detect_language_lingua(text)
        if lingua_lang and lingua_lang in SUPPORTED:
            return lingua_lang, f"{whisper_lang}+very-low-conf → Lingua={lingua_lang}"

    # Rule 4: Any remaining unsupported language → English fallback (process, don't reject)
    if whisper_lang not in SUPPORTED:
        return "en", f"{whisper_lang} unsupported → fallback English"

    return whisper_lang, "no-correction"


def _apply_english_asr_corrections(text: str) -> str:
    """Fix common Whisper English mis-recognitions for Tirumala context."""
    corrections = {
        "kirmala": "Tirumala",
        "thirumala": "Tirumala",
        "tirupathi": "Tirupati",
        "venkateshwara": "Venkateswara",
    }
    for wrong, right in corrections.items():
        text = re.sub(rf'\b{wrong}\b', right, text, flags=re.IGNORECASE)
    return text

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SAMPLE_RATE       = 16000       # Hz — required by Whisper
CHANNELS          = 1
WHISPER_MODEL_SIZE = "large-v3" # Upgraded to large-v3 for maximum accuracy in Indic dialects
WHISPER_DEVICE     = "cpu"      # "cpu" or "cuda"
WHISPER_COMPUTE    = "int8"     # int8 (fastest on CPU), float16 (GPU)

# Indic language code map: Whisper lang code → IndicTrans2 lang code
WHISPER_TO_INDICTRANS = {
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "bn": "ben_Beng",
    "ur": "urd_Arab",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "en": "eng_Latn",
}


class RealTimeSTT:
    """
    Captures audio from the default microphone on command (Push-to-Talk)
    and transcribes completed utterances with faster-whisper.

    Usage:
        stt = RealTimeSTT()
        stt.start_recording()
        # ... user speaks ...
        result = stt.stop_recording_and_transcribe()
        # result = {"text": "...", "detected_lang": "hi", "indictrans_lang": "hin_Deva"}
    """

    def __init__(self):
        logger.info("Loading Whisper model (%s)...", WHISPER_MODEL_SIZE)
        self.model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
        )
        self._stream = None
        self._frames = []

    def start_recording(self):
        """Start microphone capture into an internal buffer."""
        self._frames = []
        
        target_device = None
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                name = dev['name'].lower()
                # Prioritize soundcore microphone, ignoring output-only devices
                if dev['max_input_channels'] > 0 and 'soundcore' in name:
                    target_device = i
                    break
        except Exception as e:
            logger.warning("Failed to query devices: %s", e)
            
        if target_device is not None:
            logger.info("Auto-selected microphone device %d: %s", target_device, devices[target_device]['name'])

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            device=target_device,
            callback=self._callback
        )
        self._stream.start()
        logger.info("Microphone strictly recording (PTT active)...")

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.warning("SD status: %s", status)
        self._frames.append(indata.copy())

    def stop_recording_and_transcribe(self, language_hint: str | None = None) -> dict | None:
        """
        Stop capture, convert the buffered PCM frames, and feed to Whisper.

        Args:
            language_hint: ISO 639-1 code (e.g. "te", "hi") to force Whisper
                           to transcribe in that language. If None or "auto",
                           Whisper auto-detects the language.

        Returns the transcription dict or None if invalid.
        """
        if self._stream is None:
            return None

        self._stream.stop()
        self._stream.close()
        self._stream = None

        if not self._frames:
            return None

        raw_audio = np.concatenate(self._frames, axis=0)
        audio_np = raw_audio.astype(np.float32) / 32768.0
        audio_np = audio_np.flatten()

        dur = len(audio_np) / SAMPLE_RATE
        logger.info("Transcribing %.1f seconds of audio...", dur)
        if dur < 0.2:
            return None  # Ignore accidental lightning-fast presses

        # ── Pass 1: auto-detect language (no prompt so detector is unbiased) ──
        pass1_kwargs = {
            "beam_size": 5,
            "vad_filter": True,
            "vad_parameters": dict(min_silence_duration_ms=500),
        }
        if language_hint and language_hint != "auto":
            pass1_kwargs["language"] = language_hint
            logger.info("Whisper forced to language: %s", language_hint)

        segments, info = self.model.transcribe(audio_np, **pass1_kwargs)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if not text:
            return None

        whisper_lang = info.language  # e.g. "hi", "te", "en"
        lang = whisper_lang

        # ── Language Correction (Rules 2, 3, 4) ──────────────────────────────
        # Apply Tirumala correction policy BEFORE Lingua override:
        #   ur → hi, pa+low-conf → kn/hi, any other unsupported → en fallback
        lang, correction_reason = _correct_language(
            whisper_lang, info.language_probability, text
        )
        if correction_reason != "no-correction":
            logger.info("Lang correction: %s", correction_reason)

        # ── Script detection: highest-priority override for native-script text ──
        # Reads Unicode block ranges directly — 100% immune to Whisper acoustic
        # confusion between Kannada / Telugu / Tamil / Hindi scripts.
        # Only fires when the transcribed text actually contains non-ASCII chars.
        script_lang = _detect_script_language(text)
        if script_lang:
            if script_lang != lang:
                logger.info(
                    "Script detection override: whisper=%s (conf %.2f) → script=%s",
                    lang, info.language_probability, script_lang,
                )
            lang = script_lang
        elif info.language_probability < 0.75 or lang not in ["en", "hi", "te", "ta", "kn"]:
            # For romanized / transliterated text, fall back to Lingua
            detected = _detect_language_lingua(text)
            if detected:
                if detected != lang:
                    logger.info(
                        "Lingua override: corrected=%s (conf %.2f) → Lingua=%s",
                        lang, info.language_probability, detected,
                    )
                lang = detected

        # Rule 4: If still unsupported after all correction attempts,
        # silently process as English (don't reject the pilgrim's input).
        if lang not in ["en", "hi", "te", "ta", "kn"]:
            logger.warning(
                "Language '%s' could not be mapped; falling back to English.", lang
            )
            lang = "en"

        indictrans_lang = WHISPER_TO_INDICTRANS.get(lang, "eng_Latn")

        # ── Pass 2 (non-English): refinement with explicit language + domain prompt ──
        # Re-running with the correct language code and Tirumala-domain vocabulary
        # fixes hallucinated or mis-spelled words from the auto-detect pass.
        # SKIP when Pass 1 confidence is very high — the transcript is already reliable.
        if lang != "en" and lang in _DOMAIN_PROMPT and info.language_probability < 0.95:
            try:
                pass2_kwargs = {
                    "beam_size": 5,
                    "vad_filter": True,
                    "vad_parameters": dict(min_silence_duration_ms=500),
                    "language": lang,
                    "initial_prompt": _DOMAIN_PROMPT[lang],
                }
                ref_segs, _ = self.model.transcribe(audio_np, **pass2_kwargs)
                refined = " ".join(seg.text.strip() for seg in ref_segs).strip()
                if refined:
                    if refined != text:
                        logger.info("Refined [%s]: %r → %r", lang, text[:80], refined[:80])
                    text = refined
            except Exception as e:
                logger.warning("Refinement pass failed (%s) — using pass-1 text", e)

        # For English, re-run with domain prompt to reduce hallucinations on
        # Tirumala-specific proper nouns.
        elif lang == "en":
            try:
                pass2e_kwargs = {
                    "beam_size": 5,
                    "vad_filter": True,
                    "vad_parameters": dict(min_silence_duration_ms=500),
                    "language": "en",
                    "initial_prompt": _ENGLISH_DOMAIN_PROMPT,
                }
                ref_segs, _ = self.model.transcribe(audio_np, **pass2e_kwargs)
                refined = " ".join(seg.text.strip() for seg in ref_segs).strip()
                if refined:
                    text = refined
            except Exception as e:
                logger.warning("English refinement pass failed (%s)", e)

        # ── Pass 2.3 (English): dictionary-based ASR correction ──
        # Fixes words like "Kirmala" → "Tirumala" that Whisper consistently mishears
        if lang == "en":
            text = _apply_english_asr_corrections(text)

        # ── Pass 2.5 (non-English): Phonetic dictionary + LLM correction ──
        # Corrects known Whisper ASR substitutions token-by-token (e.g. "ఇర్వే"→"ఇరవై",
        # "తమ్ముదు"→"తొమ్మిది", "సబ్బిలు"→"సభ్యులు") then uses LLM to fix any
        # remaining phonetic errors using context — also provides an English translation
        # which avoids a separate Whisper Pass 3 in most cases.
        llm_english = ""   # set below if phonetic correction succeeds
        if lang != "en":
            try:
                from query.phonetic_corrector import correct_and_translate
                correction  = correct_and_translate(text, lang)
                corrected   = correction.get("corrected", "")
                llm_english = correction.get("english", "")
                if corrected and corrected != text:
                    text = corrected
            except Exception as e:
                logger.warning("Phonetic correction skipped: %s", e)

        # ── Pass 3 (non-English): translate to English for the RAG pipeline ──
        # LLM English from Pass 2.5 takes priority (most contextually accurate).
        # Whisper audio→translate is the fallback when LLM is unavailable.
        english_text = llm_english if llm_english else text
        if lang != "en" and not llm_english:
            try:
                pass3_kwargs = {
                    "task": "translate",
                    "beam_size": 5,
                    "vad_filter": True,
                    "vad_parameters": dict(min_silence_duration_ms=500),
                    "initial_prompt": _ENGLISH_DOMAIN_PROMPT,
                }
                en_segs, _ = self.model.transcribe(audio_np, **pass3_kwargs)
                whisper_en = " ".join(seg.text.strip() for seg in en_segs).strip()
                if whisper_en:
                    english_text = whisper_en
                    logger.info("Whisper translate [%s→en]: %s", lang, english_text[:100])
            except Exception as e:
                logger.warning("Translate pass failed (%s) — will use Google Translate fallback", e)

        result = {
            "text": text,
            "english_text": english_text,
            "detected_lang": lang,
            "indictrans_lang": indictrans_lang,
            "whisper_confidence": info.language_probability,
        }
        logger.info("STT result: %s", result)
        return result

# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    stt = RealTimeSTT()
    print("\n🎤 PTT Test. Press Ctrl+C to quit.\n")
    try:
        while True:
            input("Press ENTER to start recording...")
            stt.start_recording()
            input("Press ENTER to stop recording...")
            res = stt.stop_recording_and_transcribe()
            if res:
                print(f"\n📝 Transcribed : {res['text']}")
                print(f"🌐 Language    : {res['detected_lang']} ({res['indictrans_lang']})")
                print(f"🔍 Confidence  : {res['whisper_confidence']:.2%}\n")
    except KeyboardInterrupt:
        print("Stopped.")