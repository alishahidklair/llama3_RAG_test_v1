import sys
import shutil
from pathlib import Path
import pytest


# -------------------------------------------------
# Ensure project root is on PYTHONPATH
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.indexer import build_index


# -------------------------------------------------
# Test paths
# -------------------------------------------------
DOCS_PATH = PROJECT_ROOT / "data" / "documents"
TEST_INDEX_PATH = PROJECT_ROOT / "vectorstore" / "test_faiss_index"


# -------------------------------------------------
# Shared fixture (runs once per module)
# -------------------------------------------------
@pytest.fixture(scope="module")
def index_data():
    # Clean previous test index
    if TEST_INDEX_PATH.exists():
        shutil.rmtree(TEST_INDEX_PATH)

    docs, chunks, db = build_index(
        docs_path=DOCS_PATH,
        index_path=TEST_INDEX_PATH,
        rebuild=True
    )

    return docs, chunks, db


# -------------------------------------------------
# Tests
# -------------------------------------------------

def test_documents_loaded(index_data):
    docs, _, _ = index_data
    assert len(docs) > 0, "No documents were loaded"


def test_docx_present(index_data):
    docs, _, _ = index_data
    sources = [doc.metadata.get("source", "") for doc in docs]

    assert any(source.endswith(".docx") for source in sources), \
        "DOCX file was not loaded"


def test_chunks_created(index_data):
    _, chunks, _ = index_data
    assert len(chunks) > 0, "No chunks were created"


def test_faiss_index_size(index_data):
    _, chunks, db = index_data

    assert db.index.ntotal == len(chunks), \
        "FAISS vector count does not match number of chunks"


def test_retrieval_returns_results(index_data):
    _, _, db = index_data

    results = db.similarity_search(
        "developmental psychology research methods",
        k=5
    )

    assert len(results) > 0, \
        "Similarity search returned no results"
