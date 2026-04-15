"""
tts_module.py
=============
Text-to-Speech for the Multilingual Query Architecture.

Strategy (cascading, fully local-first):
  1. Coqui TTS  — High-quality, local neural TTS (best for English)
  2. gTTS        — Google TTS over network (good multilingual support)
  3. pyttsx3     — Fully offline fallback (system voices)

Supported Indic languages via gTTS:
  hi (Hindi), te (Telugu), ta (Tamil), kn (Kannada), ml (Malayalam),
  mr (Marathi), gu (Gujarati), pa (Punjabi), bn (Bengali)

Install dependencies:
    pip install TTS gTTS pyttsx3 pygame

Note: Coqui TTS downloads model on first use (~150 MB for the default model).
      Set USE_COQUI=False to skip it entirely and always use gTTS.
"""

import os
import io
import tempfile
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
USE_COQUI   = False   # Set True if you have a GPU or are okay with CPU latency
USE_GTTS    = True    # Requires internet; best multilingual quality
USE_PYTTSX3 = True    # Offline fallback

# gTTS language map: ISO 639-1 → gTTS lang code
GTTS_LANG_MAP = {
    "hi": "hi",  # Hindi
    "te": "te",  # Telugu
    "ta": "ta",  # Tamil
    "kn": "kn",  # Kannada
    "ml": "ml",  # Malayalam
    "mr": "mr",  # Marathi
    "gu": "gu",  # Gujarati
    "pa": "pa",  # Punjabi
    "bn": "bn",  # Bengali
    "ur": "ur",  # Urdu
    "en": "en",  # English
}


class MultilingualTTS:
    """
    Converts text to speech in the detected language.

    Usage:
        tts = MultilingualTTS()
        tts.speak("నమస్కారం, మీకు స్వాగతం!", lang="te")
        tts.speak("Hello, welcome to Tirupati!", lang="en")

        # Non-blocking (fire and forget)
        tts.speak_async("आपका स्वागत है!", lang="hi")
    """

    def __init__(self):
        self._coqui_model = None
        self._pyttsx3_engine = None
        self._lock = threading.Lock()

        if USE_COQUI:
            self._init_coqui()
        if USE_PYTTSX3:
            self._init_pyttsx3()

    # ── PUBLIC API ──────────────────────────────────────────────────────────

    def speak(self, text: str, lang: str = "en"):
        """
        Speak text in the given language (ISO 639-1 code).
        Blocks until playback completes.
        """
        if not text.strip():
            return
        lang = lang.lower()[:2]
        logger.info("TTS speak [%s]: %s", lang, text[:60])

        # gTTS first for all languages (cross-platform, reliable)
        if USE_GTTS and lang in GTTS_LANG_MAP:
            if self._speak_gtts(text, lang):
                return

        # English → try Coqui as upgrade if available
        if lang == "en" and USE_COQUI and self._coqui_model:
            if self._speak_coqui(text):
                return

        # Final fallback → pyttsx3 (offline, Windows SAPI)
        if USE_PYTTSX3 and self._pyttsx3_engine:
            self._speak_pyttsx3(text)

    def speak_async(self, text: str, lang: str = "en"):
        """Non-blocking version of speak()."""
        t = threading.Thread(target=self.speak, args=(text, lang), daemon=True)
        t.start()

    # ── BACKENDS ────────────────────────────────────────────────────────────

    def _init_coqui(self):
        try:
            from TTS.api import TTS as CoquiTTS  # type: ignore
            logger.info("Loading Coqui TTS model...")
            self._coqui_model = CoquiTTS("tts_models/en/ljspeech/tacotron2-DDC")
            logger.info("Coqui TTS ready.")
        except Exception as e:
            logger.warning("Coqui TTS init failed: %s", e)
            self._coqui_model = None

    def _speak_coqui(self, text: str) -> bool:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            self._coqui_model.tts_to_file(text=text, file_path=tmp_path)
            self._play_audio_file(tmp_path)
            os.unlink(tmp_path)
            return True
        except Exception as e:
            logger.warning("Coqui TTS failed: %s", e)
            return False

    def _speak_gtts(self, text: str, lang: str) -> bool:
        try:
            from gtts import gTTS
            gtts_lang = GTTS_LANG_MAP.get(lang, "en")
            tts_obj = gTTS(text=text, lang=gtts_lang, slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            tts_obj.save(tmp_path)
            self._play_audio_file(tmp_path)
            os.unlink(tmp_path)
            return True
        except Exception as e:
            logger.warning("gTTS failed: %s", e)
            return False

    def _init_pyttsx3(self):
        try:
            import pyttsx3
            self._pyttsx3_engine = pyttsx3.init()
            self._pyttsx3_engine.setProperty("rate", 160)
            self._pyttsx3_engine.setProperty("volume", 0.9)
            logger.info("pyttsx3 TTS ready.")
        except Exception as e:
            logger.warning("pyttsx3 init failed: %s", e)
            self._pyttsx3_engine = None

    def _speak_pyttsx3(self, text: str):
        try:
            with self._lock:
                self._pyttsx3_engine.say(text)
                self._pyttsx3_engine.runAndWait()
        except Exception as e:
            logger.error("pyttsx3 TTS failed: %s", e)

    def _play_audio_file(self, path: str):
        """Play an audio file using pygame (cross-platform)."""
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(50)
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except Exception as e:
            logger.warning("pygame playback failed (%s); trying os fallback", e)
            self._play_audio_os(path)

    @staticmethod
    def _play_audio_os(path: str):
        """OS-level fallback audio playback."""
        import platform
        system = platform.system()
        if system == "Linux":
            os.system(f"aplay '{path}' 2>/dev/null || mpg123 '{path}' 2>/dev/null")
        elif system == "Darwin":
            os.system(f"afplay '{path}'")
        elif system == "Windows":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    tts = MultilingualTTS()

    test_phrases = [
        ("en", "Welcome to the Sri Venkateswara Temple information system."),
        ("te", "తిరుమల తిరుపతి దేవస్థానాలకు స్వాగతం."),
        ("hi", "श्री वेंकटेश्वर मंदिर में आपका स्वागत है।"),
        ("ta", "திருமலை திருப்பதி தேவஸ்தானங்களுக்கு வரவேற்கிறோம்."),
        ("kn", "ಶ್ರೀ ವೆಂಕಟೇಶ್ವರ ದೇವಸ್ಥಾನಕ್ಕೆ ಸುಸ್ವಾಗತ."),
    ]

    for lang, text in test_phrases:
        print(f"\n🔊 [{lang.upper()}] {text}")
        tts.speak(text, lang=lang)