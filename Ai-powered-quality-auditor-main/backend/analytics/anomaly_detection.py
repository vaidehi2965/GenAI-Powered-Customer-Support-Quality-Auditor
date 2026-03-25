"""
Anomaly Detection Module
Detects abnormal patterns in compliance violations and conversation metrics.
Architecture Decision: Separate anomaly detection to:
  1. Support statistical analysis of violation trends
  2. Implement sliding window for spike detection
  3. Track baseline metrics per agent
  4. Enable configurable sensitivity thresholds
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


@dataclass
class AnomalyAlert:
    """Represents a detected anomaly"""
    alert_type: str  # "violation_spike", "unusual_escalation", "metric_deviation"
    severity: str  # "low", "medium", "high", "critical"
    description: str
    metric_value: float
    baseline_value: float
    deviation_percent: float
    timestamp: str
    affected_agent: Optional[str] = None
    recommendation: Optional[str] = None


class BaselineCalculator:
    """Calculate baseline metrics for comparison"""
    
    def __init__(self, window_size: int = 50):
        """
        Initialize with sliding window size.
        Window size determines how many previous items to use for baseline.
        """
        self.window_size = window_size
        self.compliance_history = deque(maxlen=window_size)
        self.escalation_history = deque(maxlen=window_size)
        self.violation_history = deque(maxlen=window_size)
    
    def add_compliance_score(self, score: float) -> None:
        """Add compliance score to history"""
        self.compliance_history.append(score)
    
    def add_escalation_risk(self, risk: float) -> None:
        """Add escalation risk to history"""
        self.escalation_history.append(risk)
    
    def add_violation_count(self, count: int) -> None:
        """Add violation count to history"""
        self.violation_history.append(count)
    
    def get_baseline_compliance(self) -> float:
        """Calculate baseline compliance score"""
        if not self.compliance_history:
            return 80.0  # Default baseline
        return statistics.mean(self.compliance_history)
    
    def get_baseline_escalation(self) -> float:
        """Calculate baseline escalation risk"""
        if not self.escalation_history:
            return 20.0  # Default baseline
        return statistics.mean(self.escalation_history)
    
    def get_baseline_violations(self) -> float:
        """Calculate baseline violation count"""
        if not self.violation_history:
            return 2.0  # Default baseline
        return statistics.mean(self.violation_history)
    
    def get_std_dev_compliance(self) -> float:
        """Get standard deviation of compliance scores"""
        if len(self.compliance_history) < 2:
            return 5.0
        return statistics.stdev(self.compliance_history)
    
    def get_std_dev_violations(self) -> float:
        """Get standard deviation of violation counts"""
        if len(self.violation_history) < 2:
            return 1.0
        return statistics.stdev(self.violation_history)


class SpikeDetector:
    """Detects spikes in compliance violations"""
    
    def __init__(self, sensitivity: float = 2.0):
        """
        Initialize spike detector.
        Sensitivity: standard deviations from mean (default 2.0 = 95% confidence)
        """
        self.sensitivity = sensitivity
        self.baseline_calc = BaselineCalculator()
    
    def detect_violation_spike(self, current_violations: int) -> Optional[AnomalyAlert]:
        """
        Detect if current violation count is anomalously high.
        Uses statistical method: z-score > sensitivity threshold
        """
        self.baseline_calc.add_violation_count(current_violations)
        
        baseline = self.baseline_calc.get_baseline_violations()
        std_dev = self.baseline_calc.get_std_dev_violations()
        
        if std_dev == 0:
            return None
        
        z_score = abs(current_violations - baseline) / std_dev
        
        if z_score > self.sensitivity:
            deviation_percent = ((current_violations - baseline) / baseline * 100) if baseline > 0 else 0
            
            if z_score > 3.0:
                severity = "critical"
            elif z_score > 2.5:
                severity = "high"
            else:
                severity = "medium"
            
            return AnomalyAlert(
                alert_type="violation_spike",
                severity=severity,
                description=f"Violation count ({current_violations}) deviates {deviation_percent:.1f}% from baseline ({baseline:.1f})",
                metric_value=float(current_violations),
                baseline_value=baseline,
                deviation_percent=deviation_percent,
                timestamp=datetime.now().isoformat(),
                recommendation=f"Review recent conversations. Violations {deviation_percent:.1f}% above normal."
            )
        
        return None
    
    def detect_compliance_drop(self, current_compliance: float) -> Optional[AnomalyAlert]:
        """Detect unexpected drop in compliance scores"""
        self.baseline_calc.add_compliance_score(current_compliance)
        
        baseline = self.baseline_calc.get_baseline_compliance()
        std_dev = self.baseline_calc.get_std_dev_compliance()
        
        if std_dev == 0:
            return None
        
        z_score = abs(current_compliance - baseline) / std_dev
        
        if z_score > self.sensitivity and current_compliance < baseline:
            deviation_percent = ((baseline - current_compliance) / baseline * 100)
            
            if z_score > 3.0:
                severity = "critical"
            elif z_score > 2.5:
                severity = "high"
            else:
                severity = "medium"
            
            return AnomalyAlert(
                alert_type="compliance_drop",
                severity=severity,
                description=f"Compliance score ({current_compliance:.1f}) has dropped {deviation_percent:.1f}% from baseline ({baseline:.1f})",
                metric_value=current_compliance,
                baseline_value=baseline,
                deviation_percent=deviation_percent,
                timestamp=datetime.now().isoformat(),
                recommendation=f"Investigate root cause. Score {deviation_percent:.1f}% below normal."
            )
        
        return None


class RiskyConversationDetector:
    """Detects high-risk conversations based on multiple signals"""
    
    def __init__(self):
        """Initialize detector"""
        self.spike_detector = SpikeDetector()
    
    def analyze_conversation(self, 
                            compliance_violations: int,
                            escalation_risk: float,
                            sentiment_score: float,
                            emotional_intensity: float) -> Dict[str, Any]:
        """
        Comprehensive high-risk conversation detection.
        Combines multiple signals: violations, escalation, sentiment, emotion.
        """
        risk_score = 0.0
        risk_factors = []
        
        # Factor 1: Compliance violations (weighted 35%)
        if compliance_violations > 4:
            risk_score += 35
            risk_factors.append(f"High violation count: {compliance_violations}")
        elif compliance_violations > 2:
            risk_score += 20
            risk_factors.append(f"Moderate violation count: {compliance_violations}")
        
        # Factor 2: Escalation risk (weighted 30%)
        if escalation_risk > 70:
            risk_score += 30
            risk_factors.append(f"Critical escalation risk: {escalation_risk:.0f}%")
        elif escalation_risk > 50:
            risk_score += 20
            risk_factors.append(f"High escalation risk: {escalation_risk:.0f}%")
        
        # Factor 3: Customer sentiment (weighted 20%)
        if sentiment_score < -0.7:
            risk_score += 20
            risk_factors.append(f"Very negative customer sentiment: {sentiment_score:.2f}")
        elif sentiment_score < -0.3:
            risk_score += 10
            risk_factors.append(f"Negative customer sentiment: {sentiment_score:.2f}")
        
        # Factor 4: Emotional intensity (weighted 15%)
        if emotional_intensity > 70:
            risk_score += 15
            risk_factors.append(f"High emotional intensity: {emotional_intensity:.0f}%")
        elif emotional_intensity > 50:
            risk_score += 8
            risk_factors.append(f"Moderate emotional intensity: {emotional_intensity:.0f}%")
        
        # Determine risk level
        if risk_score >= 80:
            risk_level = "CRITICAL"
        elif risk_score >= 60:
            risk_level = "HIGH"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "overall_risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "requires_intervention": risk_score >= 60,
            "recommended_action": self._get_recommendation(risk_level, risk_factors)
        }
    
    def _get_recommendation(self, risk_level: str, factors: List[str]) -> str:
        """Generate recommendation based on risk level"""
        if risk_level == "CRITICAL":
            return "URGENT: Escalate to supervisor. Immediate intervention required."
        elif risk_level == "HIGH":
            return "Review conversation. Prepare coaching plan for agent."
        elif risk_level == "MEDIUM":
            return "Monitor conversation. Provide agent guidance if needed."
        else:
            return "Standard monitoring. No immediate action required."


class AnomalyDetectionEngine:
    """Main anomaly detection engine"""
    
    def __init__(self):
        """Initialize engine"""
        self.spike_detector = SpikeDetector(sensitivity=2.0)
        self.conversation_analyzer = RiskyConversationDetector()
        self.alerts: List[AnomalyAlert] = []
        self.alert_history: deque = deque(maxlen=100)  # Keep last 100 alerts
    
    def process_audit_result(self,
                            audit_data: Dict[str, Any],
                            sentiment_data: Dict[str, Any] = None,
                            agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process full audit result and detect anomalies.
        Returns detection results and any alerts.
        """
        anomalies = []
        analysis_results = {}
        
        # Extract key metrics
        compliance_violations = len(audit_data.get("violations", []))
        escalation_risk = sentiment_data.get("escalation", {}).get("escalation_risk", 0) if sentiment_data else 0
        sentiment_score = sentiment_data.get("sentiment", {}).get("sentiment_score", 0) if sentiment_data else 0
        emotional_intensity = sentiment_data.get("emotion", {}).get("emotional_intensity", 0) if sentiment_data else 0
        
        # Check for violation spike
        spike_alert = self.spike_detector.detect_violation_spike(compliance_violations)
        if spike_alert:
            anomalies.append(spike_alert)
            self.alert_history.append(spike_alert)
        
        # Check for compliance drop
        compliance_score = audit_data.get("compliance", 0)
        compliance_alert = self.spike_detector.detect_compliance_drop(compliance_score)
        if compliance_alert:
            anomalies.append(compliance_alert)
            self.alert_history.append(compliance_alert)
        
        # Comprehensive conversation risk analysis
        risk_analysis = self.conversation_analyzer.analyze_conversation(
            compliance_violations, escalation_risk, sentiment_score, emotional_intensity
        )
        analysis_results["conversation_risk"] = risk_analysis
        
        return {
            "anomalies_detected": len(anomalies),
            "alerts": [asdict(a) for a in anomalies],
            "analysis": analysis_results,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts"""
        if not self.alert_history:
            return {"total_alerts": 0, "recent_alerts": []}
        
        critical_count = sum(1 for a in self.alert_history if a.severity == "critical")
        high_count = sum(1 for a in self.alert_history if a.severity == "high")
        
        return {
            "total_alerts": len(self.alert_history),
            "critical_alerts": critical_count,
            "high_alerts": high_count,
            "recent_alerts": [asdict(a) for a in list(self.alert_history)[-5:]]
        }
