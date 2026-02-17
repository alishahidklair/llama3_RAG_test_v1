from pathlib import Path
import shutil

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------
# Path handling (robust)
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = PROJECT_ROOT / "data" / "documents"
INDEX_PATH = PROJECT_ROOT / "vectorstore" / "faiss_index"


# -----------------------------
# Core indexing function
# -----------------------------

def build_index(
    docs_path: Path = DOCS_PATH,
    index_path: Path = INDEX_PATH,
    rebuild: bool = True,
):
    """
    Load documents, chunk them, embed them, and build a FAISS index.

    Returns:
        docs, chunks, db
    """

    if rebuild and index_path.exists():
        shutil.rmtree(index_path)

    docs = []

    for file in docs_path.iterdir():
        if file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
        elif file.suffix.lower() == ".docx":
            loader = Docx2txtLoader(str(file))
        else:
            continue

        loaded_docs = loader.load()
        if not loaded_docs:
            print(f"⚠️ Warning: {file.name} loaded 0 pages")

        docs.extend(loaded_docs)

    if not docs:
        raise RuntimeError("❌ No documents were loaded")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    db.save_local(str(index_path))

    return docs, chunks, db


# -----------------------------
# Script entry point
# -----------------------------

if __name__ == "__main__":
    print("🔍 Building FAISS index...")
    docs, chunks, db = build_index()

    print("✅ Indexing complete")
    print(f"📄 Documents loaded : {len(docs)}")
    print(f"🧩 Chunks created   : {len(chunks)}")
    print(f"📦 FAISS vectors    : {db.index.ntotal}")
    print(f"📁 Index saved to   : {INDEX_PATH}")
