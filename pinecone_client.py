from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

def get_pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in .env")

    pc = Pinecone(api_key=api_key)

    index_name = "call-knowledge"
    index = pc.Index(index_name)

    return index