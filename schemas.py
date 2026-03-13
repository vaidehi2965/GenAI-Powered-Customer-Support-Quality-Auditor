from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuditResult:
    file_name: str
    agent_name: str
    transcript: str
    retrieved_context: list[dict[str, Any]] = field(default_factory=list)
    quality_score: float = 0
    compliance_score: float = 0
    resolution_score: float = 0
    final_score: float = 0
    violations: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    summary: str = ""
    sentiment_label: str = "Neutral"
    sentiment_score: float = 0.0
    dominant_emotion: str = "Calm"
    compliance_alerts: list[dict[str, str]] = field(default_factory=list)
    manager_summary: str = ""
