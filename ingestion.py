from __future__ import annotations

from pathlib import Path
import pandas as pd
from docx import Document
from pypdf import PdfReader

from app.config import SUPPORTED_DOC_TYPES


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def read_csv(path: Path) -> str:
    df = pd.read_csv(path)
    return df.to_csv(index=False)


def read_xlsx(path: Path) -> str:
    xls = pd.ExcelFile(path)
    joined = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        joined.append(f"Sheet: {sheet}\n{df.to_csv(index=False)}")
    return "\n\n".join(joined)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_DOC_TYPES:
        return ""
    if suffix in {".txt", ".md"}:
        return read_text_file(path)
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    if suffix == ".csv":
        return read_csv(path)
    if suffix == ".xlsx":
        return read_xlsx(path)
    return ""
