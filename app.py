import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import streamlit as st
from database.db import SessionLocal, CallRecord
from groq_auditor import evaluate_call

st.set_page_config(page_title="Supervisor Dashboard", layout="wide")

st.title("📞 AI Call Quality Dashboard")

session = SessionLocal()

calls = session.query(CallRecord).all()

for call in calls:
    with st.expander(f"📁 {call.file_name}"):

        st.subheader("Transcript")
        st.write(call.transcript)

        if st.button(f"Evaluate {call.id}"):
            result = evaluate_call(
                call.transcript,
                "Did the agent follow company policy?"
            )
            call.feedback = result
            session.commit()

        if call.feedback:
            st.subheader("AI Evaluation")
            st.success(call.feedback)