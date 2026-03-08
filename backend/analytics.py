from __future__ import annotations

from pathlib import Path
import pandas as pd

from app.config import REPORTS_DIR
from app.utils import load_json


def load_report_rows() -> list[dict]:
    rows = []
    for path in REPORTS_DIR.glob("*.json"):
        payload = load_json(path, default={}) or {}
        if payload:
            rows.append(payload)
    return rows


def build_dashboard_dataframe() -> pd.DataFrame:
    rows = load_report_rows()
    if not rows:
        return pd.DataFrame(
            columns=[
                "file_name",
                "agent_name",
                "quality_score",
                "compliance_score",
                "resolution_score",
                "final_score",
                "violations_count",
            ]
        )
    data = []
    for row in rows:
        data.append(
            {
                "file_name": row.get("file_name"),
                "agent_name": row.get("agent_name", "Unknown"),
                "quality_score": row.get("quality_score", 0),
                "compliance_score": row.get("compliance_score", 0),
                "resolution_score": row.get("resolution_score", 0),
                "final_score": row.get("final_score", 0),
                "violations_count": len(row.get("violations", [])),
            }
        )
    return pd.DataFrame(data)
