"""
Sentiment & Emotion Detection Module
Real-time analysis of sentiment, emotions, and escalation risk.
Architecture Decision: Separate analytics into dedicated module to:
  1. Decouple sentiment analysis from quality scoring
  2. Support multiple sentiment backends (Hugging Face, TextBlob, etc.)
  3. Implement emotion classification for agent coaching
  4. Track escalation patterns over time
"""

import json
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class SentimentCategory(Enum):
    """Sentiment classifications"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class EmotionType(Enum):
    """Emotion classifications"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    FRUSTRATION = "frustration"
    CONFUSION = "confusion"


@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    overall_sentiment: str
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    sentiment_trajectory: List[float]  # Progression through conversation
    key_sentiment_phrases: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EmotionResult:
    """Result of emotion analysis"""
    primary_emotion: str
    emotion_scores: Dict[str, float]  # emotion -> score (0-100)
    emotional_intensity: float  # 0-100
    emotional_peaks: List[Tuple[int, str]]  # (position, emotion)
    emotion_transitions: List[str]  # Detected changes in emotion
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EscalationAnalysis:
    """Analysis of conversation escalation risk"""
    escalation_risk: float  # 0-100
    escalation_indicators: List[str]
    is_escalating: bool
    de_escalation_needed: bool
    customer_sentiment_trend: str  # improving, stable, declining
    agent_sentiment_trend: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SentimentAnalyzer:
    """Analyzes sentiment of conversation text"""
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        self.sentiment_keywords = {
            "very_positive": ["excellent", "amazing", "great", "love", "fantastic", "wonderful", "delighted"],
            "positive": ["good", "nice", "happy", "satisfied", "pleased", "helpful", "appreciate"],
            "neutral": ["understand", "okay", "fine", "noted", "acknowledge"],
            "negative": ["bad", "poor", "upset", "frustrated", "disappointed", "unhappy"],
            "very_negative": ["hate", "terrible", "awful", "disgusted", "furious", "worst", "unacceptable"]
        }
    
    def analyze(self, text: str, depth: int = 0) -> SentimentResult:
        """
        Analyze overall sentiment of text.
        Method: Keyword-based with LLM validation (can be upgraded with transformers)
        """
        text_lower = text.lower()
        
        # Count sentiment keywords
        scores = {}
        for sentiment, keywords in self.sentiment_keywords.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            scores[sentiment] = count
        
        # Determine overall sentiment
        if scores["very_positive"] > 0:
            overall = SentimentCategory.VERY_POSITIVE
            score = 1.0
        elif scores["positive"] > scores["negative"]:
            overall = SentimentCategory.POSITIVE
            score = 0.6
        elif scores["negative"] > scores["positive"]:
            overall = SentimentCategory.NEGATIVE
            score = -0.6
        elif scores["very_negative"] > 0:
            overall = SentimentCategory.VERY_NEGATIVE
            score = -1.0
        else:
            overall = SentimentCategory.NEUTRAL
            score = 0.0
        
        # Extract sentiment phrases
        phrases = self._extract_sentiment_phrases(text)
        
        trajectory = self._calculate_trajectory(text) if depth == 0 else []
        
        return SentimentResult(
            overall_sentiment=overall.value,
            sentiment_score=score,
            confidence=0.8,  # Can be improved with ML model
            sentiment_trajectory=trajectory,
            key_sentiment_phrases=phrases
        )
    
    def _extract_sentiment_phrases(self, text: str, window: int = 5) -> List[str]:
        """Extract phrases around sentiment keywords"""
        phrases = []
        words = text.split()
        
        for i, word in enumerate(words):
            if any(word.lower() == kw for kws in self.sentiment_keywords.values() for kw in kws):
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                phrase = " ".join(words[start:end])
                phrases.append(phrase)
        
        return phrases[:5]  # Return top 5
    
    def _calculate_trajectory(self, text: str) -> List[float]:
        """Calculate sentiment score progression through conversation"""
        chunks = [c for c in text.split(".") if c.strip()]
        if len(chunks) <= 1:
            return []
            
        trajectory = []
        for chunk in chunks:
            result = self.analyze(chunk, depth=1)
            trajectory.append(result.sentiment_score)
        
        return trajectory


class EmotionDetector:
    """Detects emotions in conversation text"""
    
    def __init__(self):
        """Initialize emotion detector"""
        self.emotion_indicators = {
            EmotionType.JOY: ["happy", "excited", "wonderful", "delighted", "thrilled", "love"],
            EmotionType.SADNESS: ["sad", "depressed", "unhappy", "miserable", "down", "lonely"],
            EmotionType.ANGER: ["angry", "furious", "enraged", "mad", "livid", "hate"],
            EmotionType.FEAR: ["scared", "afraid", "worried", "anxious", "concerned", "nervous"],
            EmotionType.SURPRISE: ["surprised", "shocked", "amazed", "wow", "unexpected"],
            EmotionType.DISGUST: ["disgusted", "repulsed", "gross", "nasty", "revolting"],
            EmotionType.FRUSTRATION: ["frustrated", "annoyed", "irritated", "exasperated"],
            EmotionType.CONFUSION: ["confused", "lost", "unclear", "don't understand", "puzzled"]
        }
    
    def analyze(self, text: str, depth: int = 0) -> EmotionResult:
        """
        Detect emotions in text.
        Returns emotional intensity and emotion distribution.
        """
        text_lower = text.lower()
        emotion_scores = {}
        
        # Score each emotion
        for emotion, indicators in self.emotion_indicators.items():
            count = sum(1 for indicator in indicators if indicator in text_lower)
            emotion_scores[emotion.value] = min(100, count * 20)  # Scale to 0-100
        
        # Find primary emotion
        primary = max(emotion_scores, key=emotion_scores.get)
        
        # Calculate overall intensity
        intensity = sum(emotion_scores.values()) / len(emotion_scores)
        
        # Only find peaks and transitions at top level to avoid infinite recursion
        peaks = self._find_peaks(text) if depth == 0 else []
        transitions = self._find_transitions(text) if depth == 0 else []
        
        return EmotionResult(
            primary_emotion=primary,
            emotion_scores=emotion_scores,
            emotional_intensity=intensity,
            emotional_peaks=peaks,
            emotion_transitions=transitions
        )
    
    def _find_peaks(self, text: str) -> List[Tuple[int, str]]:
        """Find points of maximum emotional intensity"""
        sentences = [s for s in text.split(".") if s.strip()]
        # If text has no periods, it's just one sentence. Skip to avoid redundancy.
        if len(sentences) <= 1:
            return []
            
        peaks = []
        for i, sentence in enumerate(sentences):
            result = self.analyze(sentence, depth=1)
            if result.emotional_intensity > 40:
                peaks.append((i, result.primary_emotion))
        
        return peaks[:5]
    
    def _find_transitions(self, text: str) -> List[str]:
        """Find transitions between different emotions"""
        sentences = [s for s in text.split(".") if s.strip()]
        if len(sentences) <= 1:
            return []
            
        transitions = []
        prev_emotion = None
        
        for sentence in sentences:
            result = self.analyze(sentence, depth=1)
            if prev_emotion and prev_emotion != result.primary_emotion:
                transitions.append(f"{prev_emotion} → {result.primary_emotion}")
            prev_emotion = result.primary_emotion
        
        return transitions


class EscalationDetector:
    """Detects escalation risk in conversations"""
    
    def __init__(self, sentiment_analyzer: SentimentAnalyzer, emotion_detector: EmotionDetector):
        self.sentiment_analyzer = sentiment_analyzer
        self.emotion_detector = emotion_detector
        
        self.escalation_keywords = [
            "unfair", "unacceptable", "ridiculous", "lawsuit", "complaint",
            "management", "supervisor", "escalate", "demand", "refuse"
        ]
    
    def analyze(self, text: str) -> EscalationAnalysis:
        """
        Analyze escalation risk in conversation.
        High risk: negative sentiment + escalation keywords + high intensity
        """
        sentiment = self.sentiment_analyzer.analyze(text)
        emotion = self.emotion_detector.analyze(text)
        
        text_lower = text.lower()
        escalation_score = 0.0
        indicators = []
        
        # Check sentiment
        if sentiment.sentiment_score < -0.5:
            escalation_score += 30
            indicators.append("Negative sentiment detected")
        
        # Check for escalation keywords
        keyword_count = sum(1 for kw in self.escalation_keywords if kw in text_lower)
        escalation_score += keyword_count * 15
        if keyword_count > 0:
            indicators.append(f"{keyword_count} escalation keyword(s) detected")
        
        # Check emotion intensity
        if emotion.emotional_intensity > 60:
            escalation_score += 25
            indicators.append(f"High emotional intensity: {emotion.emotional_intensity:.0f}")
        
        # Cap at 100
        escalation_score = min(100, escalation_score)
        
        # Determine trends
        customer_trend = "declining" if sentiment.sentiment_score < -0.3 else "improving" if sentiment.sentiment_score > 0.3 else "stable"
        agent_trend = "stable"  # Would be calculated from agent-only dialogue
        
        return EscalationAnalysis(
            escalation_risk=escalation_score,
            escalation_indicators=indicators,
            is_escalating=escalation_score > 50,
            de_escalation_needed=escalation_score > 60,
            customer_sentiment_trend=customer_trend,
            agent_sentiment_trend=agent_trend
        )


class SentimentEmotionAnalyzer:
    """Unified interface for sentiment and emotion analysis"""
    
    def __init__(self):
        """Initialize all sub-analyzers"""
        self.sentiment = SentimentAnalyzer()
        self.emotion = EmotionDetector()
        self.escalation = EscalationDetector(self.sentiment, self.emotion)
    
    def comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment, emotion, and escalation analysis.
        Returns JSON-serializable results.
        """
        sentiment_result = self.sentiment.analyze(text)
        emotion_result = self.emotion.analyze(text)
        escalation_result = self.escalation.analyze(text)
        
        return {
            "sentiment": sentiment_result.to_dict(),
            "emotion": emotion_result.to_dict(),
            "escalation": escalation_result.to_dict(),
            "timestamp": "2024-01-01T00:00:00Z"  # Would be actual timestamp
        }
    
    def to_json(self, text: str) -> str:
        """Return analysis as JSON string"""
        result = self.comprehensive_analysis(text)
        return json.dumps(result, indent=2, default=str)
