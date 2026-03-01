from sentence_transformers import SentenceTransformer
from rag.pinecone_client import get_pinecone_index

model = SentenceTransformer("all-MiniLM-L6-v2")
index = get_pinecone_index()

def retrieve_context(query, top_k=3):
    query_embedding = model.encode(query).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    contexts = [match["metadata"]["text"] for match in results["matches"]]

    return "\n".join(contexts)