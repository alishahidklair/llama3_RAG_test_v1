import os
import torch
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"

torch.set_num_threads(12)
torch.set_num_interop_threads(12)



INDEX_PATH = "../vectorstore/faiss_index"

# --- Use environment variable to control threads ---
os.environ["OMP_NUM_THREADS"] = "12"  # Use all logical CPU cores

# LLaMA 3-8B CPU config
llm = OllamaLLM(
    model="llama3",          # make sure you pulled it: `ollama pull llama3`
    temperature=0,
    system="""
You must answer ONLY using the provided context.
If the answer is not in the documents, say:
'I cannot find this information in the documents.'
"""
)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 4})
