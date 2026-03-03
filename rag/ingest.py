from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

file_path = "data/policies/compliance_policy.txt"

print("File exists:", os.path.exists(file_path))

loader = TextLoader(file_path)
documents = loader.load()

print("Documents loaded:", len(documents))

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

print("Chunks created:", len(docs))

if len(docs) == 0:
    print("No chunks created. Check your file content.")
    exit()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(docs, embeddings)

vectorstore.save_local("faiss_index")

print("FAISS index created successfully.")
