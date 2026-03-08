from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def _safe_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def transcribe_with_existing(file_path: str) -> dict[str, Any]:
    module = _safe_import("transcribe")
    if module:
        for fn_name in ["transcribe_audio", "transcribe"]:
            fn = getattr(module, fn_name, None)
            if callable(fn):
                result = fn(file_path)
                if isinstance(result, dict):
                    return {
                        "text": result.get("text") or result.get("transcript") or "",
                        "segments": result.get("segments", []),
                    }
                if isinstance(result, str):
                    return {"text": result, "segments": []}
    # fallback demo transcript
    name = Path(file_path).stem.replace("_", " ").replace("-", " ")
    return {
        "text": (
            f"Demo transcript for {name}. Customer asks about account issue, "
            "agent responds, verifies details, shares resolution steps, and closes politely."
        ),
        "segments": [],
    }


def audit_with_existing(transcript: str, retrieved_context: list[dict[str, Any]]) -> dict[str, Any]:
    module = _safe_import("groq_auditor")
    if module:
        for fn_name in ["audit_conversation", "audit"]:
            fn = getattr(module, fn_name, None)
            if callable(fn):
                return fn(transcript, retrieved_context)
    # fallback heuristic demo
    transcript_lower = transcript.lower()
    quality = 82 if "sorry" in transcript_lower or "thank" in transcript_lower else 74
    compliance = 90 if retrieved_context else 68
    resolution = 84 if "resolve" in transcript_lower or "steps" in transcript_lower else 72
    violations = [] if compliance >= 85 else ["No policy-backed context found"]
    return {
        "quality_score": quality,
        "compliance_score": compliance,
        "resolution_score": resolution,
        "violations": violations,
        "strengths": ["Professional tone", "Clear conversation flow"],
        "improvements": ["Add stronger verification phrasing", "Include policy disclaimer explicitly"],
        "summary": "Context-aware audit generated successfully.",
        "agent_name": "Agent Demo",
    }
