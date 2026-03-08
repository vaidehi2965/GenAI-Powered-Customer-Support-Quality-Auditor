# GenAI-Powered-Customer-Support-Quality-Auditor

📌 Overview

This project implements an AI-powered system to audit customer support calls.
The system converts audio recordings into text transcripts, evaluates call quality using an LLM scoring engine, and provides intelligent policy-aware analysis using Retrieval-Augmented Generation (RAG).

The system helps organizations automatically monitor customer support performance, policy compliance, and service quality.

🚀 Features
🎙️ Speech & Text Processing

Speech-to-Text using Whisper

Transcript Cleaning & Storage

Batch Processing of Multiple Audio Files

🤖 AI-Based Evaluation

LLM-Based Call Scoring

Empathy Detection

Compliance Verification

Resolution Quality Assessment

🧠 RAG-Powered Intelligence (Milestone 3)

Policy-aware auditing using company documents

Vector similarity search using FAISS

Local embeddings using HuggingFace

Context-aware LLM evaluation

📊 Analytics & Visualization

Interactive Streamlit Dashboard

Agent Performance Leaderboard

Sentiment Distribution Analysis

Compliance Violation Tracking

Quality Score Trends

Downloadable Audit Reports

🏗️ System Capabilities

Modular Architecture

Structured CSV & JSON Outputs

Knowledge Base Indexing

Retrieval Inspector for RAG testing

🏗 Project Structure
genai_support_auditor/
│
├── backend/
│   ├── ingestion.py              # Document indexing for RAG
│   ├── rag_engine.py             # Retrieval-Augmented Generation engine
│   ├── groq_auditor.py           # LLM scoring logic
│   └── adapters.py               # Transcription & audit adapters
│
├── frontend/
│   └── streamlit_app.py          # Interactive analytics dashboard
│
├── Datasets/                     # Input audio files
├── transcripts/                  # Generated transcripts
├── knowledge_base/               # Company policies / SOPs
├── reports/                      # Generated audit reports
│
├── transcribe.py                 # Speech-to-text module
├── requirements.txt
└── README.md

⚙️ Installation
1️⃣ Create Virtual Environment
python -m venv venv
2️⃣ Activate Environment

Windows

venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Install FFmpeg

Ensure FFmpeg is installed and added to PATH for audio processing.

▶️ How to Run
🧠 Milestone 1 — Transcription
python transcribe.py

Output:

transcripts/transcripts.csv
🧠 Milestone 2 — LLM Scoring
python groq_auditor.py

Output:

Empathy Score

Compliance Score

Quality Score

Structured Audit Results

🧠 Milestone 3 — RAG-Based Intelligent Auditing
Step 1 — Index Knowledge Base
python backend/ingestion.py

This creates a FAISS vector index from company policy documents.

Step 2 — Run RAG Engine (Testing)
python backend/rag_engine.py

Retrieves relevant policies and performs context-aware auditing.

Step 3 — Launch Analytics Dashboard
streamlit run frontend/streamlit_app.py



🧠 Milestone Breakdown
🧠 Milestone 1 — Speech Processing Pipeline

Implemented Whisper-based speech-to-text pipeline

Processed multiple customer support recordings

Generated structured transcript CSV files

🧠 Milestone 2 — AI-Based Call Evaluation

Implemented LLM-based scoring engine

Evaluated transcripts for:

Empathy

Compliance

Quality of response

Generated structured audit reports

🧠 Milestone 3 — RAG & Analytics Platform

Implemented Retrieval-Augmented Generation (RAG)

Indexed company policies using FAISS vector database

Used HuggingFace embeddings for local vector creation

Enabled policy-aware auditing

Built interactive Streamlit dashboard

Added performance analytics and visualization

Created agent leaderboards and compliance tracking



🛠 Technologies Used
🧠 AI & ML

Whisper (Speech Recognition)

Groq LLM (LLaMA-based inference)

HuggingFace Embeddings

Retrieval-Augmented Generation (RAG)

💻 Backend

Python

LangChain

FAISS Vector Database

Pandas

🎨 Frontend

Streamlit

Plotly

Custom CSS Styling

🔧 Tools

FFmpeg

Virtual Environments

📊 Outputs
📁 Generated Files

1️⃣ transcripts.csv
→ Contains converted speech transcripts

2️⃣ Audit Reports
→ Empathy, compliance, and quality scores
→ Violations & improvement suggestions

3️⃣ FAISS Index
→ Enables fast policy retrieval

4️⃣ Dashboard Analytics
→ Visual performance monitoring

📌 Future Enhancements

Real-time streaming transcription

Emotion detection from voice

Live call monitoring

Multi-language support

Cloud deployment

CRM integration

Automated agent coaching

Advanced sentiment analysis
