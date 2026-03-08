from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATASETS_DIR,
    EMBEDDING_MODEL,
    INDEX_DIR,
    KB_DIR,
    SUPPORTED_DOC_TYPES,
    TOP_K,
)
from app.utils import chunk_text, file_hash, load_json, save_json
from backend.ingestion import extract_text


class RAGEngine:
    def __init__(self) -> None:
        self.index_path = INDEX_DIR / "faiss.index"
        self.meta_path = INDEX_DIR / "metadata.pkl"
        self.state_path = INDEX_DIR / "state.json"
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.metadata: list[dict[str, Any]] = []
        self.state = load_json(self.state_path, default={}) or {}
        self._load_or_create()

    def _load_or_create(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with self.meta_path.open("rb") as f:
                self.metadata = pickle.load(f)
        else:
            dim = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(dim)
            self.metadata = []

    def _persist(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("wb") as f:
            pickle.dump(self.metadata, f)
        save_json(self.state_path, self.state)

    def _iter_source_files(self) -> list[Path]:
        files = []
        for folder in [DATASETS_DIR, KB_DIR]:
            files.extend([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_DOC_TYPES])
        return files

    def refresh_index(self) -> dict[str, int]:
        current_files = self._iter_source_files()
        current_hashes = {str(p): file_hash(p) for p in current_files}
        if current_hashes == self.state:
            return {"indexed_files": len(current_files), "new_chunks": 0}

        dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

        texts = []
        meta = []
        for path in current_files:
            text = extract_text(path)
            if not text.strip():
                continue
            chunks = chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                meta.append(
                    {
                        "source": str(path),
                        "file_name": path.name,
                        "chunk_id": f"{path.stem}-{i}",
                        "text": chunk,
                    }
                )

        if texts:
            vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            self.index.add(vectors.astype(np.float32))
            self.metadata = meta

        self.state = current_hashes
        self._persist()
        return {"indexed_files": len(current_files), "new_chunks": len(self.metadata)}

    def add_uploaded_file(self, path: Path) -> dict[str, int]:
        # uploaded docs are already stored in datasets/ or kb/, so just refresh
        return self.refresh_index()

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        distances, indices = self.index.search(query_vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(1 / (1 + dist))
            results.append(item)
        return results
