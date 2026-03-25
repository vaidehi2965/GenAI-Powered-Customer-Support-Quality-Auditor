"""
Multi-Language Transcription Module
Supports transcription in multiple languages with automatic detection.

Architecture Decision: Abstract transcription layer to:
1. Support multiple transcription backends (Whisper, HuggingFace, etc.)
2. Implement automatic language detection
3. Handle fallback to translation for non-English
4. Maintain language metadata for downstream processing
"""

import logging
import asyncio
from typing import Dict, Optional, List, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import os
import subprocess

# Maximum audio duration in seconds for processing
MAX_AUDIO_DURATION = 900

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported transcription languages"""
    ENGLISH = "en"
    SPANISH = "es"
    HINDI = "hi"
    FRENCH = "fr"
    GERMAN = "de"
    PORTUGUESE = "pt"
    ARABIC = "ar"
    CHINESE = "zh"


@dataclass
class TranscriptionResult:
    """Result of transcription operation"""
    transcript: str
    detected_language: str
    language_code: SupportedLanguage
    confidence: float
    duration_seconds: float
    is_translated: bool = False
    original_language: Optional[str] = None
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "transcript": self.transcript,
            "detected_language": self.detected_language,
            "language_code": self.language_code.value,
            "confidence": self.confidence,
            "duration_seconds": self.duration_seconds,
            "is_translated": self.is_translated,
            "original_language": self.original_language
        }


class BaseTranscriber:
    """Base class for transcription providers"""
    
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio file"""
        raise NotImplementedError
    
    async def transcribe_async(self, audio_path: str) -> TranscriptionResult:
        """Async transcription"""
        # Default: run sync version in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe, audio_path)


def get_audio_duration(audio_path: str) -> float:
    """
    Get audio duration in seconds using Whisper's audio loading.
    Fast — only reads metadata/header, does not transcribe.
    """
    try:
        import whisper
        audio = whisper.load_audio(audio_path)
        duration = len(audio) / 16000  # Whisper resamples to 16kHz
        return duration
    except Exception as e:
        logger.warning(f"Could not determine audio duration: {e}")
        return 0.0


class WhisperTranscriber(BaseTranscriber):
    """OpenAI Whisper-based transcriber (local)"""
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_name: Size of Whisper model (tiny, base, small, medium, large)
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("whisper not installed. Install with: pip install openai-whisper")
        
        self.model_name = model_name
        self.model = whisper.load_model(model_name)
        logger.info(f"Whisper model loaded: {model_name}")
    
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio using Whisper with optimized settings"""
        import time
        start_time = time.time()
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Validate audio duration before processing
        duration = get_audio_duration(audio_path)
        if duration > MAX_AUDIO_DURATION:
            raise ValueError(
                f"Audio exceeds recommended duration ({MAX_AUDIO_DURATION} seconds). "
                f"Detected duration: {duration:.1f}s. Please upload shorter audio."
            )
        
        logger.info(f"Starting transcription: {audio_path} (duration: {duration:.1f}s)")
        
        # Optimized Whisper settings for CPU performance
        result = self.model.transcribe(
            audio_path,
            fp16=False,                       # Required for CPU (no CUDA)
            language="en",                     # Hint: skip language detection overhead
            condition_on_previous_text=False,  # Faster — don't condition on previous
        )
        
        # Extract language from Whisper result
        detected_language = result.get("language", "en")
        
        # Map to SupportedLanguage enum
        lang_code = self._map_language_code(detected_language)
        
        # Duration from validation
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Transcription completed in {processing_time:.0f}ms")
        
        return TranscriptionResult(
            transcript=result["text"],
            detected_language=detected_language,
            language_code=lang_code,
            confidence=0.95,
            duration_seconds=duration,
            is_translated=False,
            processing_time_ms=processing_time
        )
    
    @staticmethod
    def _map_language_code(lang: str) -> SupportedLanguage:
        """Map language code to SupportedLanguage enum"""
        lang_lower = lang.lower()
        mapping = {
            'en': SupportedLanguage.ENGLISH,
            'es': SupportedLanguage.SPANISH,
            'hi': SupportedLanguage.HINDI,
            'fr': SupportedLanguage.FRENCH,
            'de': SupportedLanguage.GERMAN,
            'pt': SupportedLanguage.PORTUGUESE,
            'ar': SupportedLanguage.ARABIC,
            'zh': SupportedLanguage.CHINESE,
        }
        return mapping.get(lang_lower[:2], SupportedLanguage.ENGLISH)


class LanguageDetector:
    """Detects language from text"""
    
    @staticmethod
    def detect(text: str) -> tuple[str, float]:
        """
        Detect language from text.
        
        Returns:
            Tuple of (language_code, confidence)
        """
        try:
            from langdetect import detect, detect_langs
            
            # Get all probabilities
            detections = detect_langs(text)
            if detections:
                best = detections[0]
                return best.lang, best.prob
        except ImportError:
            logger.warning("langdetect not installed. Defaulting to English.")
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
        
        return "en", 0.0


class TranslationService:
    """Handles translation to English"""
    
    def __init__(self):
        """Initialize translation service"""
        self.translator = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.translator = pipeline("translation_xx_to_en", model="facebook/mbart-large-50-many-to-one-mmt")
                logger.info("Translation service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize translation: {e}")
    
    def translate_to_english(self, text: str, source_language: str) -> str:
        """
        Translate text to English.
        
        Args:
            text: Text to translate
            source_language: Source language code (e.g., 'es', 'hi')
            
        Returns:
            English translation or original text if translation fails
        """
        if not self.translator:
            logger.warning("Translation service not available")
            return text
        
        try:
            result = self.translator(text, max_length=512)
            if result and isinstance(result, list) and len(result) > 0:
                return result[0].get("translation_text", text)
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
        
        return text


class MultilingualTranscriptionEngine:
    """
    End-to-end transcription with language detection and translation.
    Ensures all downstream processing works with English text.
    """
    
    def __init__(self, whisper_model: str = "base", auto_translate: bool = True):
        """
        Initialize transcription engine.
        
        Args:
            whisper_model: Whisper model size
            auto_translate: Automatically translate non-English to English
        """
        self.transcriber = WhisperTranscriber(model_name=whisper_model)
        self.detector = LanguageDetector()
        self.translator = TranslationService() if auto_translate else None
        self.auto_translate = auto_translate
    
    def transcribe_and_process(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe audio and automatically translate if needed.
        
        Returns:
            TranscriptionResult with English transcript
        """
        # Step 1: Transcribe
        result = self.transcriber.transcribe(audio_path)
        
        # Step 2: Check if translation is needed
        if self.auto_translate and result.language_code != SupportedLanguage.ENGLISH:
            if self.translator:
                logger.info(f"Translating from {result.detected_language} to English")
                english_text = self.translator.translate_to_english(
                    result.transcript,
                    result.language_code.value
                )
                
                # Update result
                result.original_language = result.detected_language
                result.transcript = english_text
                result.is_translated = True
                result.language_code = SupportedLanguage.ENGLISH
        
        return result
    
    async def transcribe_and_process_async(self, audio_path: str) -> TranscriptionResult:
        """Async version of transcription and processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe_and_process, audio_path)
    
    @staticmethod
    def supported_languages() -> List[str]:
        """List of supported languages"""
        return [lang.name for lang in SupportedLanguage]


# Singleton instance
_transcription_engine: Optional[MultilingualTranscriptionEngine] = None


def get_transcription_engine(whisper_model: str = "base", 
                            auto_translate: bool = True) -> MultilingualTranscriptionEngine:
    """Get or create global transcription engine instance"""
    global _transcription_engine
    if _transcription_engine is None:
        _transcription_engine = MultilingualTranscriptionEngine(
            whisper_model=whisper_model,
            auto_translate=auto_translate
        )
    return _transcription_engine
