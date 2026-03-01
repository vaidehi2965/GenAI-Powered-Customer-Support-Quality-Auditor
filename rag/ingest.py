from sentence_transformers import SentenceTransformer
from rag.pinecone_client import get_pinecone_index

model = SentenceTransformer("all-MiniLM-L6-v2")
index = get_pinecone_index()

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def ingest_knowledge(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()

        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding,
            "metadata": {"text": chunk}
        })

    index.upsert(vectors)
    print("✅ Knowledge uploaded to Pinecone!")
