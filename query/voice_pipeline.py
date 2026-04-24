"""
voice_pipeline.py
=================
Glue layer that wires STT → Language Detection → IndicTrans2 Translation
→ Your existing LangGraph RAG pipeline → TTS response.

This module is intentionally pipeline-framework-agnostic.
Replace `your_langgraph_pipeline.run(query)` with your actual invocation.

Flow:
  Microphone
      │
      ▼
  RealTimeSTT  ──► {"text", "detected_lang", "indictrans_lang"}
      │
      ▼
  IndicTrans2 (if non-English)  ──► English query
      │
      ▼
  LangGraph RAG Pipeline  ──► English answer
      │
      ▼
  IndicTrans2 (if non-English)  ──► Answer in user's language
      │
      ▼
  MultilingualTTS  ──► 🔊 spoken response

Install:
    pip install faster-whisper webrtcvad sounddevice numpy langdetect gTTS pyttsx3 pygame
    pip install git+https://github.com/AI4Bharat/IndicTrans2  (or your local path)
"""

import logging
import time
from typing import Callable
import sys
import os

# Add the repository root to sys.path so import like `query.STT` works
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query.STT import RealTimeSTT
from query.TTS import MultilingualTTS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# IndicTrans2 TRANSLATION WRAPPER
# ─────────────────────────────────────────────

class IndicTranslator:
    """
    Translates between Indic languages and English using Google Translate
    via the deep-translator library (free, no API key needed).
    """

    # Whisper ISO codes → Google Translate language codes
    LANG_MAP = {
        "te": "te", "hi": "hi", "ta": "ta", "kn": "kn",
        "ml": "ml", "mr": "mr", "gu": "gu", "bn": "bn",
        "pa": "pa", "ur": "ur", "en": "en",
    }

    LANG_NAMES = {
        "te": "Telugu", "hi": "Hindi", "ta": "Tamil", "kn": "Kannada",
        "ml": "Malayalam", "mr": "Marathi", "gu": "Gujarati",
        "bn": "Bengali", "pa": "Punjabi", "ur": "Urdu", "en": "English",
    }

    def indic_to_english(self, text: str, src_lang: str) -> str:
        """
        Translate Indic language text → English.
        Returns the English translation, or the original text if translation fails.
        Sets self._last_translation_failed = True on failure so callers can check.
        """
        self._last_translation_failed = False
        iso = self._resolve_iso(src_lang)
        if iso == "en" or not text.strip() or self._translator_cls is None:
            return text

        # ── Pre-translation: Mask Protected Terms ──
        try:
            from query.protected_terms import mask_protected_terms, restore_protected_terms
            masked_text, restore_map = mask_protected_terms(text, src_lang)
        except ImportError:
            masked_text, restore_map = text, {}

        for attempt in range(2):   # 1 retry
            try:
                translated = self._translator_cls(source=iso, target="en").translate(masked_text)
                if translated and translated.strip() and translated.strip() != masked_text.strip():
                    # ── Post-translation: Restore Protected Terms ──
                    final_translated = restore_protected_terms(translated, restore_map) if restore_map else translated
                    logger.info("Translated [%s→en]: %s → %s", iso, text[:60], final_translated[:60])
                    return final_translated
                # If translated == original, translation likely silently failed
                logger.warning("Translation [%s→en] returned same text — treating as failure.", iso)
                break
            except Exception as e:
                logger.warning("Translation [%s→en] attempt %d failed: %s", iso, attempt + 1, e)
                if attempt == 0:
                    import time as _t; _t.sleep(0.5)  # brief backoff before retry
        self._last_translation_failed = True
        logger.error("Translation [%s→en] failed after retries. Returning original.", iso)
        return text

    def __init__(self):
        self._last_translation_failed = False
        try:
            from deep_translator import GoogleTranslator
            self._translator_cls = GoogleTranslator
            logger.info("IndicTranslator ready (deep-translator / Google Translate).")
        except ImportError:
            logger.warning("deep-translator not installed. Translation will be bypassed.")
            self._translator_cls = None


    def english_to_indic(self, text: str, tgt_lang: str) -> str:
        """Translate English text → target Indic language."""
        iso = self._resolve_iso(tgt_lang)
        if iso == "en" or not text.strip() or self._translator_cls is None:
            return text
        try:
            translated = self._translator_cls(source='en', target=iso).translate(text)
            logger.info("Translated [en→%s]: %s → %s", iso, text[:60], (translated or "")[:60])
            return translated or text
        except Exception as e:
            logger.error("Translation [en→%s] failed: %s. Returning English.", iso, e)
            return text

    def _resolve_iso(self, lang: str) -> str:
        """Convert ISO codes to Google code safely."""
        return self.LANG_MAP.get(lang, lang)


# ─────────────────────────────────────────────
# VOICE PIPELINE
# ─────────────────────────────────────────────

class VoicePipeline:
    """
    Real-time voice loop that integrates STT, translation, your RAG pipeline,
    and TTS into one continuous interaction.

    Args:
        rag_fn: Callable[[str], str]
            Your LangGraph pipeline function. Takes an English query string,
            returns an English answer string.
        wake_word: str | None
            Optional wake word (e.g. "tirupati"). If set, the system only
            processes utterances that contain this word.
    """

    def __init__(self, rag_fn: Callable[[str], str], wake_word: str | None = None, language_hint: str | None = None):
        self.rag_fn         = rag_fn
        self.wake_word      = wake_word.lower() if wake_word else None
        self.language_hint  = language_hint   # ISO code or None/"auto"
        self.stt            = RealTimeSTT()
        self.tts            = MultilingualTTS()
        self.translator     = IndicTranslator()
        self._running       = False

    def run(self):
        """Start the manual voice loop. Blocks until stop() is called."""
        self._running = True
        self.tts.speak("Namaskaram! Welcome to Tirumala Assistant. Ask me anything about Tirumala.", lang="en")
        logger.info("VoicePipeline running in Push-To-Talk mode. Press Ctrl+C to quit.")

        while self._running:
            # 1. PTT Mechanic: Wait for ENTER to start
            try:
                input("\nPress [ENTER] to start speaking...")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting Voice Mode...")
                break

            self.stt.start_recording()

            # Wait for ENTER to stop
            try:
                input("Recording... Press [ENTER] to stop and transcribe...")
            except (KeyboardInterrupt, EOFError):
                self.stt.stop_recording_and_transcribe()
                print("\nExiting Voice Mode...")
                break

            stt_result = self.stt.stop_recording_and_transcribe(
                language_hint=self.language_hint
            )
            if not stt_result:
                self.tts.speak(
                    "I didn't hear anything. Could you please speak again?",
                    lang="en",
                )
                print("No speech detected. Please try again.")
                continue

            user_text      = stt_result["text"]
            detected_lang  = stt_result["detected_lang"]       # e.g. "te"
            indictrans_lang = stt_result["indictrans_lang"]    # e.g. "tel_Telu"
            whisper_conf   = stt_result.get("whisper_confidence", 0.0)

            # Rule 4: STT now auto-remaps unsupported languages to English.
            # We surface a friendly notice here when that happened, but we never
            # reject the pilgrim's input outright.
            if detected_lang == "unsupported":
                # Defensive: shouldn't happen anymore, but keep as a safety net
                logger.warning("STT returned 'unsupported' — forcing English fallback")
                stt_result["detected_lang"] = "en"
                detected_lang = "en"

            lang_name = self.translator.LANG_NAMES.get(detected_lang, detected_lang)
            print(f"\n🌐 Language: {lang_name}  (confidence: {whisper_conf:.0%})")
            print(f"You said: {user_text}")
            logger.info("[USER %s] %s", detected_lang.upper(), user_text)

            # ── Low confidence: proactively ask to repeat ──
            # Track consecutive failures for escalating alternatives
            if not hasattr(self, '_consecutive_failures'):
                self._consecutive_failures = 0

            if whisper_conf < 0.40:
                self._consecutive_failures += 1
                logger.warning(
                    "Low STT confidence (%.0f%%) — asking user to repeat (failures=%d)",
                    whisper_conf * 100, self._consecutive_failures,
                )
                if self._consecutive_failures >= 3:
                    # After 3 failures, suggest typing instead
                    self.tts.speak(
                        "I'm having trouble hearing you clearly. "
                        "You could try typing your question instead. "
                        "Press Control C to switch to text mode.",
                        lang="en",
                    )
                    print("\n💡 Tip: Press Ctrl+C to switch to text input mode.")
                    self._consecutive_failures = 0
                else:
                    self.tts.speak(
                        "I'm sorry, I couldn't hear you clearly. "
                        "Could you please speak a little louder and closer to the microphone?",
                        lang="en",
                    )
                    print("⚠ Low confidence. Please speak more clearly.")
                continue

            # 2. Optional wake word filter
            if self.wake_word and self.wake_word not in user_text.lower():
                logger.debug("Wake word not found, skipping.")
                continue

            # 3. Translate to English if needed.
            # Priority: LLM/phonetic corrector english → Whisper translate task → Google Translate
            t0 = time.time()
            whisper_en = stt_result.get("english_text", "")
            if detected_lang != "en" and whisper_en and whisper_en.strip() != user_text.strip():
                # Best path: Whisper already produced an English translation
                english_query = whisper_en
                print(f"[Translated to English]: {english_query}")
            elif detected_lang != "en":
                # Fallback: Google Translate via IndicTranslator
                english_query = self.translator.indic_to_english(user_text, detected_lang)
                translation_failed = getattr(self.translator, "_last_translation_failed", False)
                if translation_failed:
                    lang_name_fb = self.translator.LANG_NAMES.get(detected_lang, detected_lang)
                    print(f"[⚠ Translation unavailable. Processing {lang_name_fb} query directly...]")
                    english_query = user_text  # pass native text; lang_hint guides LLM
                else:
                    print(f"[Translated to English]: {english_query}")
            else:
                english_query = user_text
            logger.info("[EN QUERY] %s  (%.2fs)", english_query, time.time() - t0)

            # 4. Run RAG pipeline — pass detected_lang so LLM answers in user's language
            t0 = time.time()
            try:
                # rag_fn now receives (english_query, detected_lang) via a lambda wrapper
                english_answer = self.rag_fn(english_query, detected_lang)
            except TypeError:
                # Fallback for rag_fn callables that don't accept detected_lang
                english_answer = self.rag_fn(english_query)
            except Exception as e:
                logger.error("RAG pipeline error: %s", e)
                english_answer = "I'm sorry, I could not process your request."
            logger.info("[ANSWER] %s  (%.2fs)", english_answer[:100], time.time() - t0)

            # 5. Check if the answer is a clarification request (validation_error)
            #    Speak a short, friendly version via TTS (the full message is too long to speak)
            _clarification_markers = [
                "couldn't quite catch",
                "couldn't understand",
                "couldn't clearly understand",
                "repeat your question",
            ]
            is_clarification = any(m in english_answer.lower() for m in _clarification_markers)

            if is_clarification:
                self._consecutive_failures += 1
                # Speak short version, display full version
                if self._consecutive_failures >= 2:
                    self.tts.speak(
                        "I'm still having trouble understanding. "
                        "You can try typing your question by pressing Control C. "
                        "I understand English, Telugu, Hindi, Tamil, and Kannada.",
                        lang="en",
                    )
                    print("\n💡 Tip: Press Ctrl+C to switch to text input mode.")
                    self._consecutive_failures = 0
                else:
                    self.tts.speak(
                        "I'm sorry, I couldn't understand that clearly. "
                        "Could you please repeat your question a little more slowly?",
                        lang="en",
                    )
            else:
                # Successful answer — reset failure counter
                self._consecutive_failures = 0
                self.tts.speak(english_answer, lang=detected_lang)

    def stop(self):
        self._running = False




# ─────────────────────────────────────────────
# INTEGRATION EXAMPLE
# ─────────────────────────────────────────────

def real_rag_function(english_query: str, detected_lang: str = "en") -> str:
    """
    Calls the actual LangGraph pipeline and extracts the string answer.
    Passes detected_lang so the LLM responds directly in the user's language.
    """
    from query.agents.pipeline import run_query

    try:
        result = run_query(user_input=english_query, language=detected_lang)
        if isinstance(result, dict) and "answer" in result:
            return result["answer"]
        return str(result)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"RAG Error: {e}")
        return "I am sorry, I encountered an error while processing your request."


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    pipeline = VoicePipeline(
        rag_fn=real_rag_function,
        wake_word=None,  # Set to "tirupati" to require wake word
    )

    print("\n" + "="*60)
    print("  TTD Multilingual Voice Assistant")
    print("  Speak in Telugu, Hindi, Tamil, Kannada, or English")
    print("  Press Ctrl+C to exit")
    print("="*60 + "\n")

    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.stop()
        print("\nVoice pipeline stopped.")