from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.config import DATASETS_DIR, KB_DIR, OUTPUTS_DIR, REPORTS_DIR, SUPPORTED_AUDIO_TYPES, SUPPORTED_DOC_TYPES
from app.utils import now_stamp, save_json
from backend.adapters import audit_with_existing, transcribe_with_existing
from backend.analytics import build_dashboard_dataframe
from backend.rag_engine import RAGEngine

st.set_page_config(page_title="AuditAI", layout="wide")

CUSTOM_CSS = """
<style>
.stApp { background-color: #0b1020; color: #f5f7ff; }
[data-testid="stSidebar"] { background-color: #10172a; }
.metric-card {
    background: linear-gradient(180deg,#11192f,#0d1326);
    padding: 18px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 30px rgba(0,0,0,.25);
}
.block-title { font-size: 18px; font-weight: 700; margin-bottom: 8px; }
.small-muted { opacity: .7; font-size: 13px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_rag_engine() -> RAGEngine:
    engine = RAGEngine()
    engine.refresh_index()
    return engine


def persist_uploaded_file(upload, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / upload.name
    out_path.write_bytes(upload.getbuffer())
    return out_path


def audit_audio_file(audio_path: Path, rag: RAGEngine) -> dict:
    transcript_payload = transcribe_with_existing(str(audio_path))
    transcript = transcript_payload.get("text", "")
    retrieval_query = transcript[:1500] if transcript else audio_path.stem
    context = rag.retrieve(retrieval_query)
    audit = audit_with_existing(transcript, context)
    quality = float(audit.get("quality_score", 0))
    compliance = float(audit.get("compliance_score", 0))
    resolution = float(audit.get("resolution_score", 0))
    final_score = round((quality * 0.4) + (compliance * 0.35) + (resolution * 0.25), 2)
    result = {
        "file_name": audio_path.name,
        "transcript": transcript,
        "retrieved_context": context,
        "quality_score": quality,
        "compliance_score": compliance,
        "resolution_score": resolution,
        "final_score": final_score,
        "violations": audit.get("violations", []),
        "strengths": audit.get("strengths", []),
        "improvements": audit.get("improvements", []),
        "summary": audit.get("summary", ""),
        "agent_name": audit.get("agent_name", audio_path.stem),
        "raw_response": audit,
    }
    report_path = REPORTS_DIR / f"{audio_path.stem}_{now_stamp()}.json"
    save_json(report_path, result)
    return result


def render_metrics(df: pd.DataFrame) -> None:
    total = len(df)
    avg_quality = round(df["quality_score"].mean(), 1) if total else 0
    total_violations = int(df["violations_count"].sum()) if total else 0
    resolution_rate = round(df["resolution_score"].mean(), 1) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card'><div class='small-muted'>INTERACTIONS AUDITED</div><h2>{total:,}</h2></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='small-muted'>AVG QUALITY SCORE</div><h2>{avg_quality}/100</h2></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><div class='small-muted'>COMPLIANCE VIOLATIONS</div><h2>{total_violations}</h2></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-card'><div class='small-muted'>RESOLUTION RATE</div><h2>{resolution_rate}%</h2></div>", unsafe_allow_html=True)


def render_charts(df: pd.DataFrame) -> None:
    left, right = st.columns([1.7, 1])
    with left:
        st.markdown("### Quality Score Trend")
        if not df.empty:
            trend = df.reset_index(drop=True).copy()
            trend["run"] = trend.index + 1
            fig = px.line(trend, x="run", y="quality_score", markers=True)
            fig.update_layout(template="plotly_dark", height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run some audits to see trends.")
    with right:
        st.markdown("### Agent Leaderboard")
        if not df.empty:
            top = df.groupby("agent_name", as_index=False)["final_score"].mean().sort_values("final_score", ascending=False).head(5)
            fig = px.bar(top, x="final_score", y="agent_name", orientation="h")
            fig.update_layout(template="plotly_dark", height=320, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No leaderboard yet.")

    left2, right2 = st.columns(2)
    with left2:
        st.markdown("### Score Distribution")
        if not df.empty:
            fig = px.histogram(df, x="final_score", nbins=10)
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet.")
    with right2:
        st.markdown("### Violations by File")
        if not df.empty:
            fig = px.bar(df, x="file_name", y="violations_count")
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No violation data yet.")


def render_recent_reports() -> None:
    st.markdown("### Recent Audit Reports")
    report_files = sorted(REPORTS_DIR.glob("*.json"), reverse=True)[:5]
    if not report_files:
        st.info("No reports saved yet.")
        return
    for path in report_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        with st.expander(f"{payload.get('file_name', path.name)} · score {payload.get('final_score', 0)}"):
            st.write("**Summary**")
            st.write(payload.get("summary", ""))
            st.write("**Violations**")
            st.write(payload.get("violations", []))
            st.write("**Improvements**")
            st.write(payload.get("improvements", []))
            st.write("**Retrieved Context**")
            for item in payload.get("retrieved_context", []):
                st.code(f"{item.get('file_name', 'source')} | score={round(item.get('score', 0), 3)}\n{item.get('text', '')[:500]}")


def main() -> None:
    st.title("AuditAI Dashboard")
    st.caption("RAG-powered contextual auditing for audio/chat conversations")

    rag = get_rag_engine()

    with st.sidebar:
        st.header("Ingestion & Controls")
        st.write("Datasets and knowledge files are auto-indexed for retrieval.")

        kb_uploads = st.file_uploader(
            "Upload policy / script / knowledge files",
            accept_multiple_files=True,
            type=[s.replace('.', '') for s in SUPPORTED_DOC_TYPES],
        )
        audio_uploads = st.file_uploader(
            "Upload audio files for auditing",
            accept_multiple_files=True,
            type=[s.replace('.', '') for s in SUPPORTED_AUDIO_TYPES],
        )

        if st.button("Refresh Retrieval Index", use_container_width=True):
            stats = rag.refresh_index()
            st.success(f"Indexed {stats['indexed_files']} files and {stats['new_chunks']} chunks.")

        if kb_uploads:
            for upload in kb_uploads:
                persist_uploaded_file(upload, KB_DIR)
            stats = rag.refresh_index()
            st.success(f"Knowledge files added. Indexed {stats['new_chunks']} chunks.")

        if audio_uploads:
            for upload in audio_uploads:
                persist_uploaded_file(upload, DATASETS_DIR)
            st.success("Audio files saved to datasets.")

        st.divider()
        st.write("**Auto-retrieval sources**")
        st.write(f"- {DATASETS_DIR}")
        st.write(f"- {KB_DIR}")

        if st.button("Run Audit for All Dataset Audio", use_container_width=True):
            audio_files = [p for p in DATASETS_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_TYPES]
            if not audio_files:
                st.warning("No audio files found in datasets.")
            else:
                progress = st.progress(0)
                for i, audio_file in enumerate(audio_files, start=1):
                    audit_audio_file(audio_file, rag)
                    progress.progress(i / len(audio_files))
                st.success(f"Completed {len(audio_files)} audits.")

    df = build_dashboard_dataframe()
    render_metrics(df)
    st.write("")
    render_charts(df)
    st.write("")
    render_recent_reports()

    st.write("")
    st.markdown("### Retrieval Inspector")
    query = st.text_input("Test retrieval with any query / transcript snippet")
    if query:
        results = rag.retrieve(query)
        if not results:
            st.warning("No retrieval results found. Add policy/script documents first.")
        else:
            for item in results:
                st.code(f"{item['file_name']} · score={round(item['score'], 3)}\n{item['text'][:700]}")


if __name__ == "__main__":
    main()
