"""
Agent Assist Mode
Real-time suggestions and compliance warnings for customer service agents.
Architecture Decision: Separate agent assist to:
  1. Provide real-time coaching during calls
  2. Detect missing script elements
  3. Generate contextual compliance warnings
  4. Track agent improvement over time
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class SuggestionType(Enum):
    """Types of agent suggestions"""
    EMPATHY = "empathy"
    COMPLIANCE = "compliance"
    SCRIPT_MISSING = "script_missing"
    RESOLUTION = "resolution"
    TONE = "tone"
    CLARIFICATION = "clarification"


class WarningLevel(Enum):
    """Warning severity levels"""
    INFO = "info"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AgentSuggestion:
    """Suggestion for agent improvement"""
    type: str  # SuggestionType value
    priority: int  # 1-5, 5 is highest
    message: str
    context: str  # What triggered the suggestion
    example: Optional[str] = None
    action_items: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComplianceWarning:
    """Real-time compliance warning"""
    level: str  # WarningLevel value
    category: str  # Policy, tone, script, etc.
    message: str
    policy_reference: Optional[str] = None
    immediate_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ScriptValidator:
    """Validates conversation against expected agent script elements"""
    
    def __init__(self):
        """Initialize with standard script elements"""
        self.required_elements = {
            "greeting": ["hello", "welcome", "thank you for calling"],
            "verification": ["verify", "account", "confirm", "pin", "name"],
            "empathy": ["understand", "appreciate", "grateful", "help", "sorry", "concern"],
            "resolution": ["resolved", "resolved", "issue fixed", "taken care of"],
            "closing": ["thank you", "goodbye", "pleasure", "apologize", "have a great"],
            "escalation_offer": ["supervisor", "manager", "escalate", "transfer"]
        }
        
        self.optional_elements = {
            "product_knowledge": ["feature", "benefit", "capability"],
            "customer_success": ["training", "tutorial", "resources", "guide"]
        }
    
    def validate_conversation(self, dialogue: str) -> Dict[str, Any]:
        """Validate dialogue against script requirements"""
        dialogue_lower = dialogue.lower()
        
        missing_required = []
        found_required = []
        
        # Check required elements
        for element, keywords in self.required_elements.items():
            found = any(keyword in dialogue_lower for keyword in keywords)
            if found:
                found_required.append(element)
            else:
                missing_required.append(element)
        
        # Check optional elements
        found_optional = []
        for element, keywords in self.optional_elements.items():
            found = any(keyword in dialogue_lower for keyword in keywords)
            if found:
                found_optional.append(element)
        
        completeness_score = (len(found_required) / len(self.required_elements)) * 100
        
        return {
            "completeness_score": completeness_score,
            "found_required_elements": found_required,
            "missing_required_elements": missing_required,
            "found_optional_elements": found_optional,
            "score_breakdown": {
                "required_found": len(found_required),
                "required_total": len(self.required_elements),
                "optional_found": len(found_optional),
                "optional_total": len(self.optional_elements)
            }
        }


class RealTimeSuggestionEngine:
    """Generate real-time suggestions based on conversation analysis"""
    
    def __init__(self):
        """Initialize suggestion engine"""
        self.script_validator = ScriptValidator()
        self.empathy_keywords = ["understand", "apologize", "appreciate", "help", "concern"]
        self.compliance_keywords = ["privacy", "confidential", "authorized", "verification"]
    
    def analyze_turn(self, agent_turn: str, customer_turn: str, conversation_history: str = "") -> Dict[str, Any]:
        """
        Analyze a single agent turn and generate real-time suggestions.
        
        Args:
            agent_turn: Latest agent response
            customer_turn: Latest customer query
            conversation_history: Full conversation history
        
        Returns: Suggestions, warnings, and recommendations
        """
        suggestions: List[AgentSuggestion] = []
        warnings: List[ComplianceWarning] = []
        immediate_actions = []
        
        # Check 1: Empathy analysis
        empathy_check = self._check_empathy(agent_turn, customer_turn)
        if empathy_check["suggestion"]:
            suggestions.append(AgentSuggestion(
                type=SuggestionType.EMPATHY.value,
                priority=empathy_check["priority"],
                message=empathy_check["suggestion"],
                context=empathy_check["context"],
                example=empathy_check.get("example")
            ))
        
        # Check 2: Compliance
        compliance_check = self._check_compliance(agent_turn, customer_turn)
        if compliance_check["warning"]:
            warnings.append(ComplianceWarning(
                level=compliance_check["level"],
                category=compliance_check["category"],
                message=compliance_check["warning"],
                immediate_action=compliance_check.get("immediate_action")
            ))
        
        # Check 3: Script elements
        script_check = self._check_script_elements(agent_turn)
        if script_check["missing"]:
            suggestions.append(AgentSuggestion(
                type=SuggestionType.SCRIPT_MISSING.value,
                priority=script_check["priority"],
                message=f"Consider including: {', '.join(script_check['missing'])}",
                context="Standard agent script elements strengthen customer confidence",
                action_items=script_check["action_items"]
            ))
        
        # Check 4: Resolution progress
        if "question" in customer_turn.lower() or "?" in customer_turn:
            resolution_check = self._check_resolution(agent_turn, customer_turn)
            if resolution_check["suggestion"]:
                suggestions.append(AgentSuggestion(
                    type=SuggestionType.RESOLUTION.value,
                    priority=resolution_check["priority"],
                    message=resolution_check["suggestion"],
                    context=resolution_check["context"]
                ))
        
        return {
            "suggestions": [s.to_dict() for s in suggestions],
            "warnings": [w.to_dict() for w in warnings],
            "immediate_actions": immediate_actions,
            "summary": self._generate_summary(suggestions, warnings),
            "next_steps_recommended": len(suggestions) + len(warnings) > 0
        }
    
    def _check_empathy(self, agent_turn: str, customer_turn: str) -> Dict[str, Any]:
        """Check for empathy in agent response"""
        agent_lower = agent_turn.lower()
        customer_lower = customer_turn.lower()
        
        # Check if customer expressed concern/frustration
        has_concern = any(word in customer_lower for word in ["frustrated", "angry", "concerned", "upset", "confused", "problem", "issue"])
        
        # Check if agent showed empathy
        has_empathy = any(word in agent_lower for word in self.empathy_keywords)
        
        if has_concern and not has_empathy:
            return {
                "suggestion": "Customer expressed concern - consider showing empathy to build trust",
                "context": "Empathy correlates with higher customer satisfaction",
                "priority": 4,
                "example": "I understand how that would be frustrating. Let me help you resolve this."
            }
        
        return {"suggestion": None, "priority": 0}
    
    def _check_compliance(self, agent_turn: str, customer_turn: str) -> Dict[str, Any]:
        """Check for compliance issues"""
        agent_lower = agent_turn.lower()
        customer_lower = customer_turn.lower()
        
        # Check if sensitive data is being discussed without verification
        sensitive_keywords = ["account number", "ssn", "password", "pin", "credit card"]
        has_sensitive = any(word in customer_lower for word in sensitive_keywords)
        
        # Check if agent verified customer identity
        verified = any(word in agent_lower for word in ["verified", "confirm your", "what's your", "can you verify"])
        
        if has_sensitive and not verified:
            return {
                "warning": "Sensitive data discussed - ensure customer identity is verified",
                "level": WarningLevel.CRITICAL.value,
                "category": "Security/Privacy",
                "priority": 5,
                "immediate_action": "Ask customer verification question before discussing sensitive information"
            }
        
        return {"warning": None, "level": None, "category": None}
    
    def _check_script_elements(self, agent_turn: str) -> Dict[str, Any]:
        """Check for missing standard script elements"""
        agent_lower = agent_turn.lower()
        
        missing = []
        if "?" not in agent_turn:  # No question asked
            missing.append("open-ended question to drive conversation")
        
        if not any(word in agent_lower for word in ["is there", "anything else", "can i help"]):
            missing.append("offer to help further")
        
        return {
            "missing": missing,
            "priority": 2 if missing else 0,
            "action_items": [f"Add: {m}" for m in missing]
        }
    
    def _check_resolution(self, agent_turn: str, customer_turn: str) -> Dict[str, Any]:
        """Check if resolution is being achieved"""
        agent_lower = agent_turn.lower()
        
        # Check if agent is solving the problem
        solving_words = ["done", "resolved", "fixed", "set up", "completed", "sent you", "scheduled"]
        is_solving = any(word in agent_lower for word in solving_words)
        
        # Check if customer is confused
        confused_words = ["still don't understand", "not clear", "confused", "how", "why"]
        customer_confused = any(word in customer_turn.lower() for word in confused_words)
        
        if customer_confused and not is_solving:
            return {
                "suggestion": "Customer may still be confused. Confirm understanding with a follow-up question.",
                "context": "Resolution requires customer confirmation",
                "priority": 4
            }
        
        return {"suggestion": None, "priority": 0}
    
    def _generate_summary(self, suggestions: List[AgentSuggestion], warnings: List[ComplianceWarning]) -> str:
        """Generate brief summary of suggestions and warnings"""
        if not suggestions and not warnings:
            return "On track - no immediate issues detected"
        
        parts = []
        if warnings:
            critical = sum(1 for w in warnings if w.level == "critical")
            if critical > 0:
                return f"⚠️ CRITICAL: {critical} compliance warning(s) require immediate attention"
            parts.append(f"{len(warnings)} compliance consideration(s)")
        
        if suggestions:
            high_priority = sum(1 for s in suggestions if s.priority >= 4)
            if high_priority > 0:
                parts.append(f"{high_priority} high-priority improvement(s)")
        
        return " | ".join(parts) if parts else "Good progress"


class AgentAssistManager:
    """Main interface for agent assist mode"""
    
    def __init__(self):
        """Initialize manager"""
        self.suggestion_engine = RealTimeSuggestionEngine()
        self.script_validator = ScriptValidator()
        self.suggestion_history: List[Dict[str, Any]] = []
    
    def process_turn(self, 
                    agent_message: str,
                    customer_message: str,
                    conversation_so_far: str = "") -> Dict[str, Any]:
        """
        Process a single turn in the conversation.
        Returns suggestions, warnings, and coaching tips.
        """
        analysis = self.suggestion_engine.analyze_turn(
            agent_message, customer_message, conversation_so_far
        )
        
        self.suggestion_history.append(analysis)
        
        return {
            "turn_analysis": analysis,
            "timestamp": "2024-01-01T00:00:00Z",
            "agent_guidance": self._format_guidance(analysis)
        }
    
    def _format_guidance(self, analysis: Dict[str, Any]) -> str:
        """Format analysis into agent-friendly guidance"""
        guidance = []
        
        if analysis["warnings"]:
            guidance.append("🚨 ALERT: " + analysis["warnings"][0]["message"])
        
        if analysis["suggestions"]:
            top = analysis["suggestions"][0]
            guidance.append(f"💡 Tip: {top['message']}")
        
        return " | ".join(guidance) if guidance else "Continue conversation naturally"
    
    def validate_full_conversation(self, dialogue: str) -> Dict[str, Any]:
        """Validate complete conversation against script requirements"""
        validation = self.script_validator.validate_conversation(dialogue)
        
        return {
            "script_validation": validation,
            "passed": validation["completeness_score"] >= 70,
            "coaching_areas": validation["missing_required_elements"],
            "strengths": validation["found_required_elements"]
        }
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get summary of recent agent performance"""
        if not self.suggestion_history:
            return {"conversations_analyzed": 0}
        
        total_suggestions = sum(len(h.get("suggestions", [])) for h in self.suggestion_history)
        total_warnings = sum(len(h.get("warnings", [])) for h in self.suggestion_history)
        
        return {
            "conversations_analyzed": len(self.suggestion_history),
            "total_suggestions": total_suggestions,
            "total_warnings": total_warnings,
            "avg_suggestions_per_turn": total_suggestions / max(len(self.suggestion_history), 1),
            "coaching_focus_areas": self._identify_focus_areas()
        }
    
    def _identify_focus_areas(self) -> List[str]:
        """Identify main coaching focus areas from history"""
        suggestion_types = {}
        
        for history in self.suggestion_history:
            for suggestion in history.get("suggestions", []):
                stype = suggestion.get("type", "unknown")
                suggestion_types[stype] = suggestion_types.get(stype, 0) + 1
        
        # Return top 3 focus areas
        sorted_types = sorted(suggestion_types.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_types[:3]]
