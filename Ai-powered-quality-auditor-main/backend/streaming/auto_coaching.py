"""
Auto Coaching Engine
Analyzes agent performance over time and generates personalized improvement recommendations.
Architecture Decision: Separate coaching engine to:
  1. Track agent metrics over multiple conversations
  2. Identify weak performance areas
  3. Generate personalized coaching plans
  4. Measure improvement over time
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    empathy_avg: float
    professionalism_avg: float
    compliance_score_avg: float
    resolution_rate: float
    escalation_prevention_rate: float
    script_completion_score: float
    conversation_count: int
    avg_handle_time: float
    last_conversation_date: str


@dataclass
class CoachingPlan:
    """Personalized coaching plan for agent"""
    agent_id: str
    plan_date: str
    focus_areas: List[str]
    priority_improvements: List[str]
    coaching_tips: List[str]
    success_metrics: List[str]
    target_deadline: str


class AgentPerformanceTracker:
    """Tracks individual agent performance metrics over time"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.conversations: List[Dict[str, Any]] = []
        self.metrics_history = defaultdict(list)
    
    def add_conversation(self, audit_result: Dict[str, Any], sentiment_result: Optional[Dict[str, Any]] = None) -> None:
        """Add a conversation audit result to agent history"""
        self.conversations.append({
            "timestamp": datetime.now().isoformat(),
            "audit": audit_result,
            "sentiment": sentiment_result
        })
        
        # Track individual metrics
        if "empathy" in audit_result:
            self.metrics_history["empathy"].append(audit_result.get("empathy", 0))
        if "professionalism" in audit_result:
            self.metrics_history["professionalism"].append(audit_result.get("professionalism", 0))
        if "compliance" in audit_result:
            self.metrics_history["compliance"].append(audit_result.get("compliance", 0))
    
    def get_metrics(self) -> AgentMetrics:
        """Calculate current aggregate metrics"""
        if not self.conversations:
            return AgentMetrics(0, 0, 0, 0, 0, 0, 0, 0, "N/A")
        
        empathy_scores = self.metrics_history.get("empathy", [])
        professionalism_scores = self.metrics_history.get("professionalism", [])
        compliance_scores = self.metrics_history.get("compliance", [])
        
        return AgentMetrics(
            empathy_avg=statistics.mean(empathy_scores) if empathy_scores else 0,
            professionalism_avg=statistics.mean(professionalism_scores) if professionalism_scores else 0,
            compliance_score_avg=statistics.mean(compliance_scores) if compliance_scores else 0,
            resolution_rate=self._calculate_resolution_rate(),
            escalation_prevention_rate=self._calculate_escalation_prevention(),
            script_completion_score=self._calculate_script_completion(),
            conversation_count=len(self.conversations),
            avg_handle_time=10.5,  # Would be calculated from actual data
            last_conversation_date=self.conversations[-1]["timestamp"] if self.conversations else "N/A"
        )
    
    def _calculate_resolution_rate(self) -> float:
        """Calculate % of conversations that achieved resolution"""
        if not self.conversations:
            return 0.0
        
        resolutions = sum(1 for c in self.conversations 
                         if c.get("audit", {}).get("compliance_status") == "Pass")
        return (resolutions / len(self.conversations)) * 100
    
    def _calculate_escalation_prevention(self) -> float:
        """Calculate % of conversations with low escalation risk"""
        if not self.conversations:
            return 0.0
        
        low_escalation = sum(1 for c in self.conversations 
                            if c.get("sentiment", {}).get("escalation", {}).get("escalation_risk", 100) < 30)
        return (low_escalation / len(self.conversations)) * 100
    
    def _calculate_script_completion(self) -> float:
        """Calculate average script completion score"""
        # Would be populated from actual script validation
        return 75.0
    
    def get_trend(self, metric: str, window: int = 5) -> Dict[str, Any]:
        """Get trend for specific metric over recent conversations"""
        if metric not in self.metrics_history:
            return {"trend": "no_data"}
        
        recent_values = self.metrics_history[metric][-window:]
        
        if len(recent_values) < 2:
            return {"trend": "insufficient_data", "values": recent_values}
        
        # Calculate trend
        early_avg = statistics.mean(recent_values[:len(recent_values)//2])
        late_avg = statistics.mean(recent_values[len(recent_values)//2:])
        
        if late_avg > early_avg * 1.05:
            trend = "improving"
        elif late_avg < early_avg * 0.95:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "recent_avg": statistics.mean(recent_values),
            "values": recent_values,
            "direction_change": ((late_avg - early_avg) / early_avg * 100) if early_avg > 0 else 0
        }


class PersonalizedCoachingGenerator:
    """Generates personalized coaching plans based on agent metrics"""
    
    def __init__(self):
        """Initialize coaching generator"""
        self.coaching_strategies = {
            "empathy": {
                "low": ["Use phrases like 'I understand', 'I empathize'",
                       "Ask open-ended questions about customer feelings",
                       "Acknowledge customer frustration before solving"],
                "target": 80
            },
            "professionalism": {
                "low": ["Avoid slang and casual language",
                       "Maintain calm tone when customer is frustrated",
                       "Use proper grammar and complete sentences"],
                "target": 85
            },
            "compliance": {
                "low": ["Review compliance checklist before each call",
                       "Consult policies for ambiguous situations",
                       "Document all sensitive information handling"],
                "target": 95
            },
            "resolution": {
                "low": ["Confirm issue is resolved before closing",
                       "Offer follow-up support",
                       "Document solution in customer notes"],
                "target": 90
            },
            "de_escalation": {
                "low": ["Use empathetic language",
                       "Offer solutions, not excuses",
                       "Escalate appropriately when customer demands it"],
                "target": 80
            }
        }
    
    def generate_coaching_plan(self, agent_id: str, metrics: AgentMetrics) -> CoachingPlan:
        """
        Generate personalized coaching plan based on agent metrics.
        Identifies weakest areas and provides focused coaching.
        """
        # Identify weak areas (below 75%)
        weak_areas = []
        if metrics.empathy_avg < 75:
            weak_areas.append(("empathy", metrics.empathy_avg))
        if metrics.professionalism_avg < 75:
            weak_areas.append(("professionalism", metrics.professionalism_avg))
        if metrics.compliance_score_avg < 85:
            weak_areas.append(("compliance", metrics.compliance_score_avg))
        if metrics.resolution_rate < 75:
            weak_areas.append(("resolution", metrics.resolution_rate))
        if metrics.escalation_prevention_rate < 75:
            weak_areas.append(("de_escalation", metrics.escalation_prevention_rate))
        
        # Sort by severity (lowest first)
        weak_areas.sort(key=lambda x: x[1])
        
        # Take top 3 focus areas
        focus_areas = [area[0] for area in weak_areas[:3]]
        
        # Generate coaching tips
        coaching_tips = []
        for area in focus_areas:
            if area in self.coaching_strategies:
                coaching_tips.extend(self.coaching_strategies[area]["low"])
        
        # Generate success metrics
        success_metrics = []
        for area in focus_areas:
            if area in self.coaching_strategies:
                target = self.coaching_strategies[area]["target"]
                success_metrics.append(f"{area.replace('_', ' ').title()}: Improve to {target}%")
        
        # Generate priority improvements
        priority_improvements = self._generate_priorities(focus_areas, metrics)
        
        # Set target deadline (30 days)
        target_deadline = (datetime.now() + timedelta(days=30)).isoformat()
        
        return CoachingPlan(
            agent_id=agent_id,
            plan_date=datetime.now().isoformat(),
            focus_areas=focus_areas,
            priority_improvements=priority_improvements,
            coaching_tips=coaching_tips,
            success_metrics=success_metrics,
            target_deadline=target_deadline
        )
    
    def _generate_priorities(self, focus_areas: List[str], metrics: AgentMetrics) -> List[str]:
        """Generate prioritized action items"""
        priorities = []
        
        if "compliance" in focus_areas:
            priorities.append("CRITICAL: Review all compliance policies and pass certification test")
        
        if "empathy" in focus_areas:
            priorities.append("HIGH: Complete empathy training module")
        
        if "resolution" in focus_areas:
            priorities.append("HIGH: Practice resolution closing techniques")
        
        if "de_escalation" in focus_areas:
            priorities.append("MEDIUM: Shadow high-performing agent for de-escalation tactics")
        
        return priorities


class AutoCoachingEngine:
    """Main auto-coaching engine"""
    
    def __init__(self):
        """Initialize engine"""
        self.agent_trackers: Dict[str, AgentPerformanceTracker] = {}
        self.coaching_generator = PersonalizedCoachingGenerator()
        self.coaching_plans: Dict[str, List[CoachingPlan]] = defaultdict(list)
    
    def process_audit(self, agent_id: str, audit_result: Dict[str, Any], sentiment_result: Optional[Dict[str, Any]] = None) -> None:
        """Process audit result for agent"""
        if agent_id not in self.agent_trackers:
            self.agent_trackers[agent_id] = AgentPerformanceTracker(agent_id)
        
        self.agent_trackers[agent_id].add_conversation(audit_result, sentiment_result)
    
    def generate_coaching_plan_for_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Generate coaching plan for specific agent.
        Only generates if agent has sufficient conversation history (>= 5).
        """
        if agent_id not in self.agent_trackers:
            return None
        
        tracker = self.agent_trackers[agent_id]
        
        # Need minimum conversations for meaningful analysis
        if tracker.get_metrics().conversation_count < 5:
            return {
                "status": "insufficient_data",
                "message": f"Need minimum 5 conversations. Current: {tracker.get_metrics().conversation_count}"
            }
        
        metrics = tracker.get_metrics()
        plan = self.coaching_generator.generate_coaching_plan(agent_id, metrics)
        self.coaching_plans[agent_id].append(plan)
        
        return {
            "status": "success",
            "coaching_plan": asdict(plan),
            "agent_metrics": asdict(metrics)
        }
    
    def get_agent_progress(self, agent_id: str, metric: str = "empathy") -> Dict[str, Any]:
        """Get agent progress on specific metric"""
        if agent_id not in self.agent_trackers:
            return {"status": "agent_not_found"}
        
        tracker = self.agent_trackers[agent_id]
        trend = tracker.get_trend(metric)
        
        return {
            "agent_id": agent_id,
            "metric": metric,
            "trend_data": trend,
            "current_metrics": asdict(tracker.get_metrics())
        }
    
    def list_agents_needing_coaching(self, threshold: float = 75) -> List[Dict[str, Any]]:
        """List all agents with metrics below threshold"""
        agents_needing_help = []
        
        for agent_id, tracker in self.agent_trackers.items():
            metrics = tracker.get_metrics()
            
            # Check if any major metric is below threshold
            if (metrics.empathy_avg < threshold or 
                metrics.professionalism_avg < threshold or 
                metrics.compliance_score_avg < threshold):
                
                agents_needing_help.append({
                    "agent_id": agent_id,
                    "metrics": asdict(metrics),
                    "coaching_priority": "high" if metrics.compliance_score_avg < 70 else "medium"
                })
        
        return agents_needing_help
    
    def get_team_summary(self) -> Dict[str, Any]:
        """Get summary of team coaching needs"""
        if not self.agent_trackers:
            return {"team_size": 0, "average_metrics": None}
        
        all_metrics = [t.get_metrics() for t in self.agent_trackers.values()]
        
        team_empathy = statistics.mean([m.empathy_avg for m in all_metrics if m.empathy_avg > 0])
        team_compliance = statistics.mean([m.compliance_score_avg for m in all_metrics if m.compliance_score_avg > 0])
        team_professionalism = statistics.mean([m.professionalism_avg for m in all_metrics if m.professionalism_avg > 0])
        
        return {
            "team_size": len(self.agent_trackers),
            "conversations_analyzed": sum(m.conversation_count for m in all_metrics),
            "average_metrics": {
                "empathy": team_empathy,
                "compliance": team_compliance,
                "professionalism": team_professionalism
            },
            "agents_needing_coaching": len(self.list_agents_needing_coaching()),
            "top_performer": self._identify_top_performer(),
            "average_improvement_needed": self._calculate_avg_improvement_gap(all_metrics)
        }
    
    def _identify_top_performer(self) -> Optional[str]:
        """Identify top performing agent"""
        if not self.agent_trackers:
            return None
        
        best_agent = max(
            self.agent_trackers.items(),
            key=lambda x: x[1].get_metrics().compliance_score_avg
        )
        return best_agent[0]
    
    def _calculate_avg_improvement_gap(self, all_metrics: List[AgentMetrics]) -> float:
        """Calculate average gap between current and target performance"""
        gaps = []
        target = 85
        
        for metrics in all_metrics:
            avg_score = (metrics.empathy_avg + metrics.professionalism_avg + metrics.compliance_score_avg) / 3
            gap = max(0, target - avg_score)
            gaps.append(gap)
        
        return statistics.mean(gaps) if gaps else 0
