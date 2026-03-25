# AI Quality Auditor

**Version:** 2.0.0 | **Status:** Production-Ready

A comprehensive, enterprise-grade AI-powered quality auditing system for analyzing customer service interactions, evaluating compliance, and providing actionable insights for improvement.

## Overview

The AI Quality Auditor processes customer service interactions with advanced AI capabilities, ensuring real-time quality scoring, comprehensive compliance validation, and secure data handling. Designed with a streaming architecture, the system offers instant feedback and coaching to customer service agents while maintaining rigorous privacy standards.

## Key Features

- **PII Masking & Data Protection**: Automatically detects and masks sensitive data (credit cards, SSNs, phone numbers, emails) using regex patterns and spaCy NER prior to LLM or RAG processing.
- **Multi-Language Transcription**: Integrates Whisper-based transcription with automatic language detection across 20+ languages and translates non-English speech to English.
- **Real-Time Streaming Audit**: Features a non-blocking WebSocket architecture for live transcript processing, incremental scoring, and instantaneous compliance alerts.
- **RAG Compliance Validation**: Evaluates conversation semantics against compliance policies using Pinecone/FAISS vector databases.
- **Sentiment & Emotion Analysis**: Continuously tracks text sentiment, emotion labels, and emotional intensity for anomaly detection.
- **Agent Coaching Insights**: Delivers real-time and post-call actionable coaching recommendations based on detailed performance analytics.

## System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                       FRONTEND LAYER                        │
│ • Streamlit Dashboard & Real-time WebSocket Viewers         │
├─────────────────────────────────────────────────────────────┤
│                    FASTAPI GATEWAY LAYER                    │
│ • REST API Endpoints & WebSocket Handler (/ws/realtime)     │
├─────────────────────────────────────────────────────────────┤
│                PREPROCESSING & SECURITY LAYER               │
│ • PII Masking Engine & Multi-Language Translation System    │
├─────────────────────────────────────────────────────────────┤
│                     CORE PROCESSING LAYER                   │
│ • LLM Integration (Groq/Ollama) & Vector RAG Compliance     │
├─────────────────────────────────────────────────────────────┤
│                  ANALYTICS & INSIGHTS LAYER                 │
│ • Sentiment Analysis, Anomaly Detection & Streaming Engine  │
├─────────────────────────────────────────────────────────────┤
│                    OUTPUT & STORAGE LAYER                   │
│ • JSON/CSV Exports, Audit Logging & Performance Monitoring  │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```text
Ai quality auditor/
├── backend/                  # Backend services and logic
│   ├── api/                  # FastAPI router & WebSocket definitions
│   ├── core/                 # Core logic (LLMs, RAG, PII, Audio)
│   ├── analytics/            # Sentiment & anomaly analysis modules
│   ├── streaming/            # Real-time orchestration & coaching
│   └── auditor_service.py    # Main service orchestrator
├── frontend/                 # User interfaces (Streamlit dashboards)
├── data/                     # Data files, transcripts, audit results
├── examples/                 # Example scripts for integrations
├── .env.example              # Environment variables template
├── requirements.txt          # Python project dependencies
├── QUICKSTART.md             # Detailed quick start guide
└── ARCHITECTURE.md           # Extensive architecture documentation
```

## Installation

### Prerequisites
- Python 3.9+ (3.11 recommended)
- Minimum 8GB RAM (16GB recommended)
- FFmpeg installed (required for audio transcription)

### Setup Steps
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "Ai quality auditor"
   ```
2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## Running the System

### Option 1: Local Server
1. **Configure Environment Variables:**
   ```bash
   cp .env.example .env
   # Edit .env to add your API keys (e.g., GROQ_API_KEY)
   ```
2. **Start the API Server:**
   ```bash
   cd backend
   python -m uvicorn api.main:app --reload --port 8000
   ```
3. **Start the Dashboard:**
   **(In a new terminal instance)**
   ```bash
   streamlit run frontend/new_dashboard.py
   ```

### Option 2: Docker
```bash
docker-compose up --build
```

## API Endpoints

- **Health Checks & Info**:
  - `GET /health`
  - `GET /`
- **Transcription**:
  - `POST /transcribe`: Process audio files (MP3, WAV, etc.) with language detection.
- **PII Masking**:
  - `POST /pii/mask`: Redact sensitive information from text payloads.
- **Batch Audit Analysis**:
  - `POST /audit/batch`: Submit a complete conversation transcript for full analysis.
- **Real-Time Streaming Audits**:
  - `POST /audit/realtime/start`: Initialize a live session.
  - `WebSocket /ws/realtime`: Stream real-time utterance data.
  - `POST /audit/realtime/end`: Terminate an active live session.

## Configuration

Control system behavior by editing `.env` or exporting environment variables:

- **API Configuration**: `API_PORT`, `API_HOST`
- **LLM Settings**: `GROQ_API_KEY`, `LLM_MODEL` (e.g., `llama-3.3-70b-versatile`), `OLLAMA_API_URL`
- **RAG Integration**: `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`
- **Toggle Feature Flags**: `ENABLE_PII_MASKING`, `ENABLE_NER`, `AUTO_TRANSLATE`

## Security & Compliance

- **Data Protection**: Stringent PII masking ensures no sensitive data is passed to external LLM APIs, Vector DBs, or saved in plain text logs.
- **Regulatory Protocol**: Designed to meet strict GDPR and CCPA standard workflows.
- **Observability**: Implements comprehensive audit logging of all system operations, continuous health checks, and secure error recovery frameworks.

## Performance

*Typical performance benchmarked on standard mid-range hardware:*

- **Transcription Processing**: ~30-60 seconds for 10 minutes of audio
- **PII Masking**: ~10,000 characters per second (avg. 50-100ms per transcript)
- **Real-Time LLM Scoring**: ~1 evaluation segment per second
- **Semantical RAG Search**: 200-500ms latency
- **Complete Batch Audit**: 45-120 seconds for a full 15-minute interaction

## Roadmap

- Implement multi-agent distributed processing capabilities.
- Enhance agent coaching with ML-based trend recommendations.
- Deploy custom policy templates and CRM ecosystem integrations.
- Roll out advanced voice metric analysis (including tone, clarity, and pacing).
- Support GPU hardware acceleration for faster transcription workflows.
- Introduce optimized embedding cache structures and streaming JSON responses.
