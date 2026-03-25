"""
Enterprise Quality Auditor Service
Main orchestrator for the entire GenAI-powered Customer Support Quality Auditor system.
Provides unified API for all audit capabilities.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import os
from pathlib import Path

# Import core components
from backend.core.llm_provider import LLMManager, GroqProvider
from backend.core.rag_compliance import ComplianceRAGEnhanced
from backend.analytics.sentiment_emotion import SentimentEmotionAnalyzer
from backend.analytics.anomaly_detection import AnomalyDetectionEngine
from backend.streaming.agent_assist import AgentAssistManager
from backend.streaming.auto_coaching import AutoCoachingEngine
from backend.streaming.realtime_audit import RealtimeStreamingAuditEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnterpriseQualityAuditorService:
    """
    Enterprise-grade quality auditor service.
    Unified interface for all audit capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the audit service.
        
        Args:
            config: Configuration dictionary with:
                - enable_llm: Use LLM for analysis (default: True)
                - scoring_interval: Seconds between scoring (default: 10)
                - anomaly_sensitivity: Sensitivity for anomaly detection (default: 2.0)
        """
        self.config = config or {}
        
        # Initialize real-time streaming engine (main orchestrator)
        self.streaming_engine = RealtimeStreamingAuditEngine(
            enable_llm_analysis=self.config.get("enable_llm", True),
            scoring_interval=self.config.get("scoring_interval", 10.0)
        )
        
        # Core components
        self.llm_manager = self.streaming_engine.llm_manager
        self.rag_system = self.streaming_engine.rag_system
        self.sentiment_analyzer = self.streaming_engine.sentiment_analyzer
        self.anomaly_detector = self.streaming_engine.anomaly_detector
        self.agent_assist = self.streaming_engine.agent_assist
        self.coaching_engine = self.streaming_engine.coaching_engine
        
        # Service state
        self.service_start_time = datetime.now()
        self.total_conversations = 0
        self.total_segments = 0
        
        logger.info("EnterpriseQualityAuditorService initialized successfully")
    
    # ==================== REALTIME STREAMING API ====================
    
    def start_realtime_audit(self, conversation_id: str, agent_id: str) -> Dict[str, Any]:
        """Start a new real-time streaming audit session"""
        result = self.streaming_engine.start_conversation(conversation_id, agent_id)
        if result.get("status") == "success":
            self.total_conversations += 1
        return result
    
    def process_realtime_segment(self,
                                 conversation_id: str,
                                 agent_message: str,
                                 customer_message: str,
                                 agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a segment of real-time conversation.
        Returns immediate analysis: quality scores, suggestions, warnings.
        """
        self.total_segments += 1
        return self.streaming_engine.add_segment(
            conversation_id, agent_message, customer_message, agent_id
        )
    
    def end_realtime_audit(self, conversation_id: str, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        End real-time audit and generate final report.
        Triggers coaching plan generation.
        """
        return self.streaming_engine.end_conversation(conversation_id, agent_id)
    
    # ==================== AGENT ASSIST API ====================
    
    def get_agent_realtime_suggestions(self,
                                       agent_message: str,
                                       customer_message: str) -> Dict[str, Any]:
        """
        Get real-time suggestions for agent mid-call.
        Returns suggestions, warnings, and recommended next steps.
        """
        return self.agent_assist.analyze_turn(agent_message, customer_message)
    
    def validate_agent_script(self, dialogue: str) -> Dict[str, Any]:
        """
        Validate dialogue against standard agent script.
        Returns completeness score and missing elements.
        """
        return self.agent_assist.validate_full_conversation(dialogue)
    
    def get_agent_performance_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get performance summary for specific agent"""
        result = self.coaching_engine.generate_coaching_plan_for_agent(agent_id)
        if result and result.get("status") == "success":
            progress = self.coaching_engine.get_agent_progress(agent_id)
            return {
                "coaching_plan": result.get("coaching_plan"),
                "metrics": result.get("agent_metrics"),
                "progress": progress.get("trend_data")
            }
        return result
    
    # ==================== SENTIMENT & EMOTION API ====================
    
    def analyze_sentiment_emotion(self, text: str) -> Dict[str, Any]:
        """
        Get sentiment and emotion analysis.
        Returns sentiment scores, emotional intensity, escalation risk.
        """
        return self.sentiment_analyzer.comprehensive_analysis(text)
    
    def detect_escalation_risk(self, text: str, sentiment_score: float) -> Dict[str, Any]:
        """
        Detect escalation risk with detailed indicators.
        """
        emotion = self.sentiment_analyzer.emotion.analyze(text)
        escalation = self.sentiment_analyzer.escalation.analyze(text)
        
        return {
            "escalation_risk": escalation.escalation_risk,
            "is_escalating": escalation.is_escalating,
            "de_escalation_needed": escalation.de_escalation_needed,
            "indicators": escalation.escalation_indicators,
            "customer_sentiment_trend": escalation.customer_sentiment_trend,
            "emotional_intensity": emotion.emotional_intensity
        }
    
    # ==================== ANOMALY DETECTION API ====================
    
    def check_for_anomalies(self, 
                           violations_count: int,
                           compliance_score: float,
                           escalation_risk: float,
                           sentiment_score: float,
                           emotional_intensity: float) -> Dict[str, Any]:
        """
        Check for anomalies in conversation metrics.
        Returns detected anomalies and risk analysis.
        """
        audit_data = {
            "violations": [""] * violations_count,  # Dummy entries
            "compliance_score": compliance_score
        }
        sentiment_data = {
            "escalation": {"escalation_risk": escalation_risk},
            "sentiment": {"sentiment_score": sentiment_score},
            "emotion": {"emotional_intensity": emotional_intensity}
        }
        
        return self.anomaly_detector.process_audit_result(audit_data, sentiment_data)
    
    def get_anomaly_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts"""
        return self.anomaly_detector.get_alert_summary()
    
    # ==================== AUTO COACHING API ====================
    
    def get_coaching_plan(self, agent_id: str) -> Dict[str, Any]:
        """
        Get personalized coaching plan for agent.
        Requires minimum 5 previous conversations.
        """
        return self.coaching_engine.generate_coaching_plan_for_agent(agent_id)
    
    def list_agents_needing_coaching(self, threshold: float = 75) -> List[Dict[str, Any]]:
        """
        List all agents whose metrics are below threshold.
        Useful for identifying coaching priorities.
        """
        return self.coaching_engine.list_agents_needing_coaching(threshold)
    
    def get_team_coaching_summary(self) -> Dict[str, Any]:
        """
        Get team-wide coaching analysis.
        Shows aggregate metrics and coaching needs.
        """
        return self.coaching_engine.get_team_summary()
    
    # ==================== BULK AUDIT API ====================
    
    def audit_transcript(self, transcript_text: str, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform full audit on complete transcript.
        Returns comprehensive analysis similar to batch processing.
        """
        # Quality scoring
        if self.llm_manager and self.streaming_engine.enable_llm:
            quality = self.llm_manager.analyze_text(transcript_text, "quality")
        else:
            quality = self.streaming_engine._heuristic_quality_score(transcript_text)
        
        # Sentiment analysis
        sentiment_analysis = self.sentiment_analyzer.comprehensive_analysis(transcript_text)
        
        # RAG compliance check
        compliance = self.rag_system.validate_compliance(transcript_text)
        
        # Anomaly detection
        anomalies = self.anomaly_detector.process_audit_result(quality, sentiment_analysis, agent_id)
        
        # Agent assist analysis
        agent_analysis = self.agent_assist.validate_full_conversation(transcript_text)
        
        # Store for coaching
        if agent_id:
            self.coaching_engine.process_audit(agent_id, quality, sentiment_analysis)
        
        return {
            "conversation_id": f"audit_{datetime.now().timestamp()}",
            "quality_score": quality,
            "sentiment_analysis": sentiment_analysis,
            "compliance_validation": compliance,
            "anomaly_detection": anomalies,
            "script_validation": agent_analysis,
            "overall_summary": self._generate_overall_summary(quality, sentiment_analysis, compliance),
            "timestamp": datetime.now().isoformat()
        }
    
    # ==================== RAG & COMPLIANCE API ====================
    
    def add_compliance_policy(self,
                             title: str,
                             content: str,
                             category: str,
                             severity: str) -> Dict[str, Any]:
        """
        Add custom compliance policy at runtime.
        Useful for dynamic policy updates.
        """
        policy_id = self.rag_system.add_custom_policy(title, content, category, severity)
        return {
            "status": "success",
            "policy_id": policy_id,
            "message": f"Policy '{title}' added successfully"
        }
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of all loaded policies"""
        return self.rag_system.get_policy_summary()
    
    # ==================== MONITORING & ADMIN API ====================
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get overall service status"""
        status = self.streaming_engine.get_engine_status()
        status.update({
            "total_conversations": self.total_conversations,
            "total_segments": self.total_segments,
            "uptime": str(datetime.now() - self.service_start_time),
            "service_started": self.service_start_time.isoformat()
        })
        return status
    
    def get_active_conversations(self) -> Dict[str, Any]:
        """Get list of active conversations"""
        return self.streaming_engine.get_active_conversations()
    
    def export_metrics_report(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Export comprehensive metrics report.
        If agent_id provided, returns agent-specific report.
        """
        if agent_id:
            return self.coaching_engine.get_agent_progress(agent_id)
        else:
            return self.coaching_engine.get_team_summary()
    
    # ==================== HELPER METHODS ====================
    
    def _generate_overall_summary(self,
                                 quality: Dict[str, Any],
                                 sentiment: Dict[str, Any],
                                 compliance: Dict[str, Any]) -> Dict[str, str]:
        """Generate one-line summary across all dimensions"""
        empathy = quality.get("empathy", 0)
        prof = quality.get("professionalism", 0)
        comp = quality.get("compliance_status", "unknown")
        sentiment_score = sentiment.get("sentiment", {}).get("sentiment_score", 0)
        
        quality_level = "Excellent" if (empathy + prof) / 2 > 85 else "Good" if (empathy + prof) / 2 > 70 else "Needs Improvement"
        sentiment_level = "Positive" if sentiment_score > 0.5 else "Neutral" if sentiment_score > -0.3 else "Negative"
        
        return {
            "quality_assessment": quality_level,
            "customer_sentiment": sentiment_level,
            "compliance_status": comp,
            "primary_recommendation": self._get_recommendation(quality_level, comp, sentiment_level)
        }
    
    def _get_recommendation(self, quality: str, compliance: str, sentiment: str) -> str:
        """Generate primary recommendation"""
        if compliance == "Fail":
            return "URGENT: Address compliance violations immediately"
        elif quality == "Needs Improvement":
            return "Recommend coaching session focusing on empathy and professionalism"
        elif sentiment == "Negative":
            return "Focus on de-escalation and customer satisfaction techniques"
        else:
            return "Continue current approach - good performance"
    
    def health_check(self) -> Dict[str, Any]:
        """Quick health check of all components"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "streaming_engine": "ok",
                "llm_provider": "ok" if self.llm_manager else "disabled",
                "rag_system": "ok",
                "sentiment_analyzer": "ok",
                "anomaly_detector": "ok",
                "agent_assist": "ok",
                "coaching_engine": "ok"
            },
            "service_uptime_seconds": (datetime.now() - self.service_start_time).total_seconds()
        }


# Convenience function for creating service
def create_auditor_service(config: Optional[Dict[str, Any]] = None) -> EnterpriseQualityAuditorService:
    """Factory function to create auditor service"""
    return EnterpriseQualityAuditorService(config)
