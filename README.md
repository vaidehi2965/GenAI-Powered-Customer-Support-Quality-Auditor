# GenAI-Powered-Customer-Support-Quality-Auditor

## 📌 Overview
This project implements an AI-powered system to audit customer support calls. 
The system converts audio recordings into text transcripts using Whisper 
and evaluates call quality using an LLM scoring engine.

---

## 🚀 Features

- Speech-to-Text using Whisper
- Transcript Cleaning & Storage
- LLM-Based Call Scoring
- Structured CSV Output
- Modular Architecture

---

## 🏗 Project Structure

genai_support_auditor/
│
├── Datasets/                 # Input audio files
├── transcripts/              # Generated transcripts
    groq_auditor.py            # LLM scoring                  
├── transcribe.py             # Speech-to-text module                
├── requirements.txt
└── README.md

---

## ⚙️ Installation

1. Create virtual environment:

   python -m venv venv

2. Activate environment:

   Windows:
   venv\Scripts\activate

3. Install dependencies:

   pip install -r requirements.txt

4. Ensure FFmpeg is installed and added to PATH.

---

## ▶️ How to Run

### Step 1: Transcription

python transcribe.py

This generates:
transcripts/transcripts.csv

### Step 2: LLM Scoring

python groq_auditor.py

This generates:
emapthy
compliance
Quality scores

---

## 🧠 Milestone 1
- Implemented Whisper-based speech-to-text pipeline
- Processed multiple audio files
- Generated structured transcript CSV

## 🧠 Milestone 2
- Implemented LLM-based scoring engine
- Evaluated transcripts for empathy, compliance, and quality score
- Generated structured audit report

---

## 🛠 Technologies Used

- Python
- Whisper
- OpenAI GPT
- Pandas
- FFmpeg

---

## 📊 Output

1. transcripts.csv – Contains converted speech text
2.LLM scoring - empathy,compliance and quality scores based on agent's behaviour or tone

---

## 📌 Future Enhancements

- Sentiment analysis
- Real-time streaming transcription
- Dashboard visualization
- RAG-based compliance verification

