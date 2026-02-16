import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DOCS_PATH = "../data/documents"
INDEX_PATH = "../vectorstore/faiss_index"

# Load PDFs
docs = []
for file in os.listdir(DOCS_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DOCS_PATH, file))
        docs.extend(loader.load())

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS index
db = FAISS.from_documents(chunks, embeddings)
db.save_local(INDEX_PATH)

print("✅ Documents indexed successfully")
