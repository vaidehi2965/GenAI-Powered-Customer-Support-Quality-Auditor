"""
PII (Personally Identifiable Information) Masking Layer
Detects and masks sensitive information before processing by LLM/RAG systems.

Architecture Decision: Keep PII masking as a preprocessing layer to:
1. Prevent PII leakage to external LLMs and vector databases
2. Maintain audit trails without exposing sensitive data
3. Support compliance requirements (GDPR, CCPA, HIPAA)
4. Provide optional encryption for original transcripts
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)


class PiiType(Enum):
    """Types of PII that can be detected and masked"""
    CREDIT_CARD = "REDACTED_CARD"
    SSN = "REDACTED_SSN"
    PHONE = "REDACTED_PHONE"
    EMAIL = "REDACTED_EMAIL"
    NAME = "REDACTED_NAME"
    LOCATION = "REDACTED_LOCATION"
    DATE = "REDACTED_DATE"


@dataclass
class PiiMatch:
    """Represents a detected PII item"""
    type: PiiType
    original_value: str
    start_pos: int
    end_pos: int
    confidence: float
    replacement: str = field(default="")
    
    def __post_init__(self):
        if not self.replacement:
            self.replacement = f"[{self.type.value}]"


@dataclass
class MaskingResult:
    """Result of PII masking operation"""
    masked_text: str
    original_text: str
    detected_pii: List[PiiMatch] = field(default_factory=list)
    pii_count: int = 0
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "masked_text": self.masked_text,
            "pii_detected": len(self.detected_pii),
            "pii_types": {pii.type.name: pii.replacement for pii in self.detected_pii},
            "processing_time_ms": self.processing_time_ms
        }


class PiiMasker:
    """
    Detects and masks PII using regex patterns and spaCy NER.
    Non-blocking, thread-safe masking operations.
    """
    
    # Regex patterns for structured data
    PATTERNS = {
        PiiType.CREDIT_CARD: re.compile(
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{16}\b'
        ),
        PiiType.SSN: re.compile(
            r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b'
        ),
        PiiType.PHONE: re.compile(
            r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
        ),
        PiiType.EMAIL: re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ),
    }
    
    def __init__(self, enable_ner: bool = True, spacy_model: str = "en_core_web_sm"):
        """
        Initialize PII masker.
        
        Args:
            enable_ner: Whether to use spaCy for NER (name/location detection)
            spacy_model: spaCy model to use for NER
        """
        self.enable_ner = enable_ner
        self.nlp = None
        
        if enable_ner and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
                logger.info(f"spaCy model loaded: {spacy_model}")
            except OSError:
                logger.warning(f"spaCy model {spacy_model} not found. Install with: "
                              f"python -m spacy download {spacy_model}")
                self.nlp = None
        elif enable_ner and not SPACY_AVAILABLE:
            logger.warning("spaCy not installed. NER features disabled. "
                          "Install with: pip install spacy")
    
    def mask(self, text: str, preserve_punctuation: bool = True) -> MaskingResult:
        """
        Mask PII in text using regex and NER.
        
        Args:
            text: Input text to mask
            preserve_punctuation: Keep original punctuation positions
            
        Returns:
            MaskingResult with masked text and detected PII items
        """
        import time
        start_time = time.time()
        
        original_text = text
        masked_text = text
        detected_pii: List[PiiMatch] = []
        
        # Step 1: Regex-based masking (structured data)
        masked_text, regex_pii = self._mask_with_regex(masked_text)
        detected_pii.extend(regex_pii)
        
        # Step 2: NER-based masking (names, locations)
        if self.nlp:
            masked_text, ner_pii = self._mask_with_ner(masked_text)
            detected_pii.extend(ner_pii)
        
        processing_time = (time.time() - start_time) * 1000
        
        return MaskingResult(
            masked_text=masked_text,
            original_text=original_text,
            detected_pii=detected_pii,
            pii_count=len(detected_pii),
            processing_time_ms=processing_time
        )
    
    def _mask_with_regex(self, text: str) -> Tuple[str, List[PiiMatch]]:
        """Apply regex-based masking for structured data"""
        detected = []
        
        for pii_type, pattern in self.PATTERNS.items():
            matches = list(pattern.finditer(text))
            
            # Process matches in reverse order to preserve positions
            for match in reversed(matches):
                pii = PiiMatch(
                    type=pii_type,
                    original_value=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.95  # High confidence for regex matches
                )
                detected.append(pii)
                
                # Replace in text
                text = text[:match.start()] + pii.replacement + text[match.end():]
        
        return text, detected
    
    def _mask_with_ner(self, text: str) -> Tuple[str, List[PiiMatch]]:
        """Apply spaCy NER for name and location detection"""
        if not self.nlp:
            return text, []
        
        detected = []
        doc = self.nlp(text)
        
        # Process entities in reverse order to preserve positions
        for ent in reversed(doc.ents):
            if ent.label_ == "PERSON":
                pii_type = PiiType.NAME
                confidence = 0.85
            elif ent.label_ in ["GPE", "LOC"]:
                pii_type = PiiType.LOCATION
                confidence = 0.80
            else:
                continue  # Skip other entity types
            
            pii = PiiMatch(
                type=pii_type,
                original_value=ent.text,
                start_pos=ent.start_char,
                end_pos=ent.end_char,
                confidence=confidence
            )
            detected.append(pii)
            
            # Replace in text
            text = text[:ent.start_char] + pii.replacement + text[ent.end_char:]
        
        return text, detected
    
    def get_pii_summary(self, result: MaskingResult) -> Dict[str, int]:
        """Return count of each PII type detected"""
        summary = {}
        for pii in result.detected_pii:
            key = pii.type.name
            summary[key] = summary.get(key, 0) + 1
        return summary


class PiiMaskingPipeline:
    """
    Orchestrates PII masking across entire processing pipeline.
    Ensures PII is masked before LLM/RAG processing.
    """
    
    def __init__(self, enable_ner: bool = True, keep_mapping: bool = False):
        """
        Initialize masking pipeline.
        
        Args:
            enable_ner: Enable spaCy NER for name detection
            keep_mapping: Keep PII->replacement mapping in memory (for audit purposes only)
        """
        self.masker = PiiMasker(enable_ner=enable_ner)
        self.keep_mapping = keep_mapping
        self.masked_items: Dict[str, MaskingResult] = {}
    
    def process_transcript(self, transcript_id: str, transcript: str) -> MaskingResult:
        """
        Mask PII in transcript before processing.
        
        Args:
            transcript_id: Unique identifier for transcript
            transcript: Raw transcript text
            
        Returns:
            MaskingResult with masked text for downstream processing
        """
        result = self.masker.mask(transcript)
        
        if self.keep_mapping:
            self.masked_items[transcript_id] = result
            logger.info(f"Transcript {transcript_id}: {result.pii_count} PII items masked")
        
        return result
    
    def process_for_llm(self, transcript: str) -> str:
        """Get masked text safe for LLM processing"""
        result = self.masker.mask(transcript)
        return result.masked_text
    
    def process_for_rag(self, transcript: str) -> str:
        """Get masked text safe for RAG/vector DB processing"""
        result = self.masker.mask(transcript)
        return result.masked_text
    
    def process_for_storage(self, transcript_id: str, transcript: str, 
                           encrypt_original: bool = False) -> Dict[str, Any]:
        """
        Process transcript for storage.
        Supports optional encryption of original PII.
        
        Args:
            transcript_id: Unique ID for tracking
            transcript: Original transcript
            encrypt_original: Whether to encrypt original (not implemented - use external service)
            
        Returns:
            Dictionary with masked text and metadata
        """
        result = self.masker.mask(transcript)
        
        storage_record = {
            "transcript_id": transcript_id,
            "masked_text": result.masked_text,
            "pii_summary": self.masker.get_pii_summary(result),
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": result.processing_time_ms
        }
        
        if encrypt_original:
            storage_record["note"] = "Original PII should be encrypted separately using HSM/KMS"
        
        return storage_record
    
    def get_audit_log(self) -> Dict[str, Any]:
        """Get audit log of masking operations"""
        return {
            "total_transcripts_masked": len(self.masked_items),
            "total_pii_detected": sum(r.pii_count for r in self.masked_items.values()),
            "timestamp": datetime.now().isoformat()
        }


# Singleton instance for pipeline
_masking_pipeline: Optional[PiiMaskingPipeline] = None


def get_masking_pipeline(enable_ner: bool = True) -> PiiMaskingPipeline:
    """Get or create global masking pipeline instance"""
    global _masking_pipeline
    if _masking_pipeline is None:
        _masking_pipeline = PiiMaskingPipeline(enable_ner=enable_ner)
    return _masking_pipeline
