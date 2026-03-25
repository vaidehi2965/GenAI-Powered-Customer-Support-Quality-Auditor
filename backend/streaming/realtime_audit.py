"""
Real-Time Streaming Audit Engine
Orchestrates all modules for real-time conversation analysis and compliance monitoring.
Architecture Decision: Central orchestrator that:
  1. Handles streaming transcript input
  2. Triggers incremental scoring
  3. Owns real-time alerts and suggestions
  4. Coordinates all sub-systems
"""

import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque
from datetime import datetime
import time

# Import all sub-systems
from backend.core.llm_provider import LLMManager, GroqProvider
from backend.core.rag_compliance import ComplianceRAGEnhanced
from backend.analytics.sentiment_emotion import SentimentEmotionAnalyzer
from backend.analytics.anomaly_detection import AnomalyDetectionEngine
from backend.streaming.agent_assist import AgentAssistManager
from backend.streaming.auto_coaching import AutoCoachingEngine

logger = logging.getLogger(__name__)


@dataclass
class StreamingSegment:
    """A segment of streaming conversation"""
    segment_id: str
    agent_text: str
    customer_text: str
    timestamp: str
    duration: float  # seconds


@dataclass
class RealtimeAuditResult:
    """Result from real-time audit of a segment"""
    segment_id: str
    quality_score: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]
    agent_suggestions: Dict[str, Any]
    compliance_warnings: List[Dict[str, Any]]
    anomalies: Dict[str, Any]
    timestamp: str


class RealtimeStreamingAuditEngine:
    """
    Enterprise-grade real-time streaming audit engine.
    Processesconversation segments and performs incremental scoring.
    """
    
    def __init__(self, 
                 enable_llm_analysis: bool = True,
                 scoring_interval: float = 10.0):
        """
        Initialize streaming audit engine.
        
        Args:
            enable_llm_analysis: Use LLM for quality scoring (expensive, slower)
            scoring_interval: Seconds between quality re-scoring
        """
        # Core analyzers
        self.llm_manager = LLMManager() if enable_llm_analysis else None
        self.rag_system = ComplianceRAGEnhanced()
        self.sentiment_analyzer = SentimentEmotionAnalyzer()
        self.anomaly_detector = AnomalyDetectionEngine()
        self.agent_assist = AgentAssistManager()
        self.coaching_engine = AutoCoachingEngine()
        
        # Configuration
        self.scoring_interval = scoring_interval
        self.enable_llm = enable_llm_analysis
        
        # State management
        self.active_conversations: Dict[str, List[StreamingSegment]] = {}
        self.audit_results: deque = deque(maxlen=100)  # Keep last 100 results
        self.alerts: deque = deque(maxlen=500)  # Keep last 500 alerts
        self.segment_counter = 0
        
        # Callbacks for real-time notifications
        self.alert_callbacks: List[Callable] = []
        self.suggestion_callbacks: List[Callable] = []
    
    def start_conversation(self, conversation_id: str, agent_id: str) -> Dict[str, Any]:
        """Start tracking a new streaming conversation"""
        if conversation_id in self.active_conversations:
            logger.warning(f"Conversation {conversation_id} already active")
            return {"status": "error", "message": "Conversation already active"}
        
        self.active_conversations[conversation_id] = []
        
        logger.info(f"Started tracking conversation {conversation_id} for agent {agent_id}")
        return {
            "status": "success",
            "conversation_id": conversation_id,
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def add_segment(self, 
                   conversation_id: str,
                   agent_text: str,
                   customer_text: str,
                   agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a new segment to active conversation.
        Triggers incremental scoring every scoring_interval.
        """
        if conversation_id not in self.active_conversations:
            return {"status": "error", "message": "Conversation not found"}
        
        # Create segment
        self.segment_counter += 1
        segment = StreamingSegment(
            segment_id=f"{conversation_id}_{self.segment_counter}",
            agent_text=agent_text,
            customer_text=customer_text,
            timestamp=datetime.now().isoformat(),
            duration=0.0
        )
        
        self.active_conversations[conversation_id].append(segment)
        
        # Perform real-time analysis on this segment
        analysis_result = self._analyze_segment(segment, agent_id)
        
        # Store result
        self.audit_results.append(analysis_result)
        
        # Trigger callbacks for alerts
        # Trigger callbacks for alerts
        if analysis_result.compliance_warnings:
            for warning in analysis_result.compliance_warnings:
                self._trigger_alert_callback(warning)
        
        # Trigger callbacks for suggestions
        if analysis_result.agent_suggestions and analysis_result.agent_suggestions.get("suggestions"):
            for suggestion in analysis_result.agent_suggestions.get("suggestions", []):
                self._trigger_suggestion_callback(suggestion)
        
        return {
            "status": "success",
            "segment_id": segment.segment_id,
            "analysis": self._format_for_response(analysis_result)
        }
    
    def _analyze_segment(self, segment: StreamingSegment, agent_id: Optional[str] = None) -> RealtimeAuditResult:
        """Perform comprehensive analysis on a segment"""
        
        # Combined text for analysis
        full_text = f"Agent: {segment.agent_text}\nCustomer: {segment.customer_text}"
        
        # 1. Quality Scoring (fast local scoring, optional LLM)
        quality_score = self._score_quality(full_text)
        
        # 2. Sentiment & Emotion Analysis
        sentiment_analysis = self.sentiment_analyzer.comprehensive_analysis(full_text)
        
        # 3. Agent Assist suggestions
        agent_assist_result = self.agent_assist.process_turn(
            segment.agent_text, segment.customer_text, full_text
        )
        agent_suggestions = agent_assist_result.get("turn_analysis", {})
        
        # 4. Compliance Warnings (RAG-based)
        rag_result = self.rag_system.validate_compliance(full_text)
        compliance_warnings = self._format_compliance_warnings(rag_result)
        
        # 5. Anomaly Detection
        anomalies = self.anomaly_detector.process_audit_result(
            quality_score, sentiment_analysis, agent_id
        )
        
        # Store for coaching analysis
        if agent_id:
            self.coaching_engine.process_audit(agent_id, quality_score, sentiment_analysis)
        
        return RealtimeAuditResult(
            segment_id=segment.segment_id,
            quality_score=quality_score,
            sentiment_analysis=sentiment_analysis,
            agent_suggestions=agent_suggestions,
            compliance_warnings=compliance_warnings,
            anomalies=anomalies,
            timestamp=datetime.now().isoformat()
        )
    
    def _score_quality(self, text: str) -> Dict[str, Any]:
        """Score conversation quality (fast local scoring)"""
        # Use LLM if enabled, otherwise use fast heuristic scoring
        
        if self.llm_manager and self.enable_llm:
            try:
                result = self.llm_manager.analyze_text(text, "quality")
                return result
            except Exception as e:
                logger.warning(f"LLM analysis failed, using fallback: {e}")
        
        # Fallback: Fast heuristic scoring
        return self._heuristic_quality_score(text)
    
    def _heuristic_quality_score(self, text: str) -> Dict[str, Any]:
        """
        Keyword-based quality scoring computed dynamically from transcript.
        No hardcoded scores — all values derived from conversation content.
        """
        text_lower = text.lower()
        words = text_lower.split()
        word_count = max(len(words), 1)
        
        # ── Empathy scoring ──
        empathy_keywords = {
            "understand": 15, "sorry": 15, "apologize": 15, "appreciate": 12,
            "concern": 12, "frustrating": 10, "difficult": 10, "feel": 8,
            "hear you": 15, "completely": 8, "absolutely": 8, "i see": 10,
            "that must": 12, "let me help": 15, "i can imagine": 12,
        }
        empathy_score = 0
        empathy_matches = []
        for phrase, weight in empathy_keywords.items():
            if phrase in text_lower:
                empathy_score += weight
                empathy_matches.append(phrase)
        # Base score from text length (longer = more conversation = more opportunity)
        empathy_base = min(30, word_count // 5)
        empathy = min(100, empathy_base + empathy_score)
        
        # ── Professionalism scoring ──
        prof_keywords = {
            "hello": 12, "please": 10, "thank you": 12, "thanks": 8,
            "assist": 10, "welcome": 10, "certainly": 10, "happy to": 10,
            "of course": 8, "good morning": 12, "good afternoon": 12,
            "good evening": 12, "sir": 8, "ma'am": 8, "madam": 8,
        }
        prof_score = 0
        prof_matches = []
        for phrase, weight in prof_keywords.items():
            if phrase in text_lower:
                prof_score += weight
                prof_matches.append(phrase)
        # Penalty for unprofessional markers
        if "!" in text and text.count("!") > 2:
            prof_score -= 10
        prof_base = min(30, word_count // 5)
        professionalism = min(100, max(0, prof_base + prof_score))
        
        # ── Resolution scoring ──
        resolution_keywords = {
            "i will": 12, "let me": 12, "resolve": 15, "fix": 12,
            "investigate": 12, "look into": 12, "process": 10, "update": 8,
            "solution": 15, "here's what": 15, "going to": 10, "follow up": 12,
            "taken care": 12, "next step": 12, "action": 8, "arrange": 10,
        }
        resolution_score = 0
        resolution_matches = []
        for phrase, weight in resolution_keywords.items():
            if phrase in text_lower:
                resolution_score += weight
                resolution_matches.append(phrase)
        resolution_base = min(25, word_count // 6)
        resolution = min(100, resolution_base + resolution_score)
        
        # ── Compliance scoring ──
        compliance_keywords = {
            "verify": 15, "confirm": 12, "policy": 15, "account number": 12,
            "security check": 15, "for your safety": 12, "disclaimer": 10,
            "terms and conditions": 12, "regulation": 10, "compliance": 12,
            "identity": 10, "authorization": 12, "agreed": 8, "recorded": 10,
        }
        compliance_score = 0
        compliance_matches = []
        for phrase, weight in compliance_keywords.items():
            if phrase in text_lower:
                compliance_score += weight
                compliance_matches.append(phrase)
        compliance_base = min(30, word_count // 5)
        compliance_total = min(100, compliance_base + compliance_score)
        
        # Compliance violations
        violations = []
        compliance_status = "pass"
        if "refund" in text_lower and "policy" not in text_lower and "process" not in text_lower:
            violations.append("Refund discussed without policy reference")
            compliance_status = "warn"
        if "personal" in text_lower and "verify" not in text_lower and "confirm" not in text_lower:
            violations.append("Personal data mentioned without verification step")
            compliance_status = "warn"
        if violations:
            compliance_total = max(0, compliance_total - len(violations) * 10)
        
        # ── Escalation risk ──
        escalation_keywords = [
            "angry", "frustrated", "upset", "complaint", "unacceptable",
            "ridiculous", "worst", "terrible", "outraged", "disgusted",
            "furious", "never again", "cancel", "lawyer", "sue",
            "supervisor", "manager", "escalate",
        ]
        escalation_hits = sum(1 for kw in escalation_keywords if kw in text_lower)
        escalation_risk = min(100, escalation_hits * 15)
        
        # ── Strengths and recommendations ──
        strengths = []
        if empathy >= 60:
            strengths.append("Good empathetic language")
        if professionalism >= 60:
            strengths.append("Professional tone maintained")
        if resolution >= 60:
            strengths.append("Active problem-solving approach")
        if compliance_total >= 70:
            strengths.append("Compliance procedures followed")
        
        recommendations = []
        if empathy < 60:
            recommendations.append("Use more empathetic phrases like 'I understand' or 'I'm sorry'")
        if professionalism < 60:
            recommendations.append("Add professional greetings and polite language")
        if resolution < 60:
            recommendations.append("Offer specific solutions with clear next steps")
        if compliance_total < 70:
            recommendations.append("Reference policies and confirm verification steps")
        if escalation_risk > 30:
            recommendations.append("Customer shows frustration — prioritize de-escalation")
        
        return {
            "empathy": empathy,
            "professionalism": professionalism,
            "resolution": resolution,
            "compliance_status": compliance_status,
            "compliance": compliance_total,
            "escalation_risk": escalation_risk,
            "violations": violations,
            "key_issues": [f"Escalation risk: {escalation_risk}%"] if escalation_risk > 30 else [],
            "strengths": strengths if strengths else ["Conversation in progress"],
            "recommendations": [r for r in recommendations if r],
        }
    
    def _format_compliance_warnings(self, rag_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert RAG results to compliance warnings"""
        warnings = []
        
        for violation in rag_result.get("violations", []):
            warnings.append({
                "level": "warning" if violation["severity"] == "medium" else violation["severity"],
                "policy": violation["policy"],
                "message": f"Potential {violation['severity']} violation: {violation['policy']}",
                "guidance": violation.get("guidance", ""),
                "relevance_score": violation["relevance_score"]
            })
        
        return warnings
    
    def end_conversation(self, conversation_id: str, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        End conversation and generate final audit report.
        Triggers coaching analysis if agent_id provided.
        """
        if conversation_id not in self.active_conversations:
            return {"status": "error", "message": "Conversation not found"}
        
        segments = self.active_conversations.pop(conversation_id)
        
        if not segments:
            return {"status": "error", "message": "No segments in conversation"}
        
        # Generate final report
        final_report = self._generate_final_report(conversation_id, segments)
        
        # Trigger coaching plan generation if needed
        coaching_plan = None
        if agent_id:
            coaching_result = self.coaching_engine.generate_coaching_plan_for_agent(agent_id)
            coaching_plan = coaching_result.get("coaching_plan") if coaching_result else None
        
        return {
            "status": "success",
            "conversation_id": conversation_id,
            "final_report": final_report,
            "coaching_plan": coaching_plan,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_final_report(self, conversation_id: str, segments: List[StreamingSegment]) -> Dict[str, Any]:
        """Generate comprehensive final audit report"""
        
        # Aggregate metrics from all segments
        all_results = [r for r in self.audit_results if r.segment_id.startswith(conversation_id)]
        
        if not all_results:
            return {"error": "No results available"}
        
        # Calculate aggregates
        empathy_scores = [r.quality_score.get("empathy", 0) for r in all_results]
        professionalism_scores = [r.quality_score.get("professionalism", 0) for r in all_results]
        resolution_scores = [r.quality_score.get("resolution", 0) for r in all_results]
        compliance_scores = [r.quality_score.get("compliance", 0) for r in all_results]
        
        avg_empathy = sum(empathy_scores) / len(empathy_scores) if empathy_scores else 0
        avg_professionalism = sum(professionalism_scores) / len(professionalism_scores) if professionalism_scores else 0
        avg_resolution = sum(resolution_scores) / len(resolution_scores) if resolution_scores else 0
        avg_compliance = sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0
        
        # Collect all violations
        all_violations = []
        for result in all_results:
            all_violations.extend(result.quality_score.get("violations", []))
        
        return {
            "conversation_id": conversation_id,
            "segments_analyzed": len(segments),
            "metrics": {
                "empathy_avg": avg_empathy,
                "professionalism_avg": avg_professionalism,
                "compliance_avg": avg_compliance,
                "resolution_avg": avg_resolution
            },
            "violations_found": len(set(all_violations)),
            "compliance_status": "PASS" if avg_compliance > 80 else "WARN" if avg_compliance > 60 else "FAIL",
            "total_warnings": sum(len(r.compliance_warnings) for r in all_results),
            "total_suggestions": sum(len(r.agent_suggestions.get("suggestions", [])) for r in all_results),
            "segment_timeline": [
                {
                    "segment": s.segment_id,
                    "compliance_score": r.quality_score.get("compliance", 0),
                    "timestamp": r.timestamp
                }
                for s, r in zip(segments, all_results)
            ]
        }
    
    def register_alert_callback(self, callback: Callable) -> None:
        """Register callback for compliance alerts"""
        self.alert_callbacks.append(callback)
    
    def register_suggestion_callback(self, callback: Callable) -> None:
        """Register callback for agent suggestions"""
        self.suggestion_callbacks.append(callback)
    
    def _trigger_alert_callback(self, alert: Dict[str, Any]) -> None:
        """Trigger all registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def _trigger_suggestion_callback(self, suggestion: Dict[str, Any]) -> None:
        """Trigger all registered suggestion callbacks"""
        for callback in self.suggestion_callbacks:
            try:
                callback(suggestion)
            except Exception as e:
                logger.error(f"Error in suggestion callback: {e}")
    
    def _format_for_response(self, result: RealtimeAuditResult) -> Dict[str, Any]:
        """Format analysis result for API response"""
        return {
            "quality": result.quality_score,
            "sentiment": result.sentiment_analysis,
            "suggestions": result.agent_suggestions,
            "warnings": result.compliance_warnings,
            "anomalies": result.anomalies
        }
    
    def get_active_conversations(self) -> Dict[str, Any]:
        """Get list of active conversations"""
        return {
            "active_count": len(self.active_conversations),
            "conversations": [
                {
                    "conversation_id": cid,
                    "segments": len(segments)
                }
                for cid, segments in self.active_conversations.items()
            ]
        }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get overall engine status"""
        return {
            "status": "operational",
            "active_conversations": len(self.active_conversations),
            "total_alerts": len(self.alerts),
            "total_results": len(self.audit_results),
            "components": {
                "llm_enabled": self.enable_llm,
                "rag_system": "operational",
                "sentiment_analyzer": "operational",
                "anomaly_detector": "operational",
                "agent_assist": "operational",
                "coaching_engine": "operational"
            },
            "timestamp": datetime.now().isoformat()
        }
