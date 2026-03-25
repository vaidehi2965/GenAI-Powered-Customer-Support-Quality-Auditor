"""
LLM Provider Layer
Abstraction for large language models (Groq, OpenAI, etc.)
Decouples LLM implementation from business logic.
Architecture Decision: Separate LLM calls into a provider class to:
  1. Enable easy switching between LLM providers
  2. Add caching and retry logic
  3. Implement cost tracking
  4. Support fallback mechanisms
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Interface for all LLM providers"""
    
    @abstractmethod
    def query(self, prompt: str, response_format: Optional[str] = None) -> str:
        """Send query to LLM and get response"""
        pass
    
    @abstractmethod
    def query_with_context(self, prompt: str, context: str, response_format: Optional[str] = None) -> str:
        """Send query with additional context"""
        pass


class GroqProvider(BaseLLMProvider):
    """Groq LLM Provider - Production ready with error handling"""
    
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize Groq client
        Args:
            model: Model name (default: Llama 3.3 70B for best accuracy)
        """
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.max_retries = 3
    
    def query(self, prompt: str, response_format: Optional[str] = None) -> str:
        """Execute LLM query with retry logic"""
        return self._execute_query(prompt, response_format)
    
    def query_with_context(self, prompt: str, context: str, response_format: Optional[str] = None) -> str:
        """Execute LLM query with context"""
        full_prompt = f"{context}\n\n{prompt}"
        return self._execute_query(full_prompt, response_format)
    
    def _execute_query(self, prompt: str, response_format: Optional[str] = None) -> str:
        """Internal execution with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                }
                
                if response_format == "json":
                    kwargs["response_format"] = {"type": "json_object"}
                
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                
                logger.debug(f"LLM query successful on attempt {attempt + 1}")
                return content
                
            except Exception as e:
                logger.warning(f"LLM query failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"LLM query failed after {self.max_retries} attempts") from e
        
        return ""


class LLMCache:
    """Simple in-memory cache for LLM responses (can be upgraded to Redis)"""
    
    def __init__(self):
        self.cache: Dict[str, str] = {}
    
    def get(self, key: str) -> Optional[str]:
        return self.cache.get(key)
    
    def set(self, key: str, value: str) -> None:
        self.cache[key] = value
    
    def clear(self) -> None:
        self.cache.clear()


class LLMManager:
    """Manager for LLM interactions with caching and logging"""
    
    def __init__(self, provider: Optional[BaseLLMProvider] = None, use_cache: bool = True):
        self.provider = provider or GroqProvider()
        self.cache = LLMCache() if use_cache else None
    
    def analyze_text(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """
        Analyze text with different analysis types
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis (quality, sentiment, compliance, etc.)
        
        Returns: JSON response from LLM
        """
        cache_key = f"{analysis_type}:{hash(text) % 100000}" if self.cache else None
        
        if self.cache and cache_key in self.cache.cache:
            logger.debug(f"Cache hit for {analysis_type}")
            return json.loads(self.cache.get(cache_key))
        
        prompt = self._get_analysis_prompt(text, analysis_type)
        response = self.provider.query(prompt, response_format="json")
        
        try:
            result = json.loads(response)
            if self.cache and cache_key:
                self.cache.set(cache_key, response)
            return result
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {response}")
            return {"error": "Invalid JSON response", "raw": response}
    
    def _get_analysis_prompt(self, text: str, analysis_type: str) -> str:
        """Build prompt based on analysis type"""
        
        if analysis_type == "quality":
            return f"""Analyze this customer service conversation deeply and return JSON:
{text}

Return JSON ONLY:
{{
  "empathy": 0-100,
  "professionalism": 0-100,
  "resolution": 0-100,
  "compliance_status": "pass|warn|fail",
  "compliance": 0-100,
  "escalation_risk": 0-100,
  "key_issues": ["List critical issues"],
  "strengths": ["List strong points"],
  "recommendations": ["Actionable improvements"]
}}"""
        
        elif analysis_type == "sentiment":
            return f"""Analyze sentiment and emotional tone:
{text}

Return JSON ONLY:
{{
  "overall_sentiment": "positive|neutral|negative",
  "sentiment_score": -1.0 to 1.0,
  "emotion_detected": ["joy", "anger", "frustration", "satisfaction"],
  "escalation_risk": 0-100,
  "emotional_intensity": 0-100
}}"""
        
        elif analysis_type == "compliance":
            return f"""Evaluate compliance issues in this transcript:
{text}

Return JSON ONLY:
{{
  "compliance_violations": ["List violations"],
  "risk_level": "low|medium|high|critical",
  "regulatory_concerns": ["List concerns"],
  "required_actions": ["List actions needed"],
  "severity_score": 0-100
}}"""
        
        return ""
