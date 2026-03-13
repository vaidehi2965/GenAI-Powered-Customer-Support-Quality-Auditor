from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets"
KB_DIR = BASE_DIR / "knowledge_base"
OUTPUTS_DIR = BASE_DIR / "outputs"
INDEX_DIR = OUTPUTS_DIR / "index"
REPORTS_DIR = OUTPUTS_DIR / "reports"

SUPPORTED_DOC_TYPES = {".txt", ".md", ".pdf", ".docx", ".csv", ".xlsx"}
SUPPORTED_AUDIO_TYPES = {".wav", ".mp3", ".m4a", ".aac", ".ogg"}

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss").lower()  # faiss | pinecone
TOP_K = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "auditai-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

for folder in [DATASETS_DIR, KB_DIR, OUTPUTS_DIR, INDEX_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
