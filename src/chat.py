from rag_chain import llm, retriever
import gradio as gr
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def answer_query(query: str) -> str:
    """
    Handles a single user query using the RAG retriever and LLaMA LLM.
    Returns the response text.
    """

    # Use the previously working call
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Use ONLY the context below to answer the question.
If the answer is not in the context, say:
'I cannot find this information in the documents.'

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    # Optionally append sources
    if docs:
        sources = [Path(doc.metadata.get("source", "Unknown")).name for doc in docs]
        response += "\n\nSources: " + ", ".join(sources)

    return response



# -----------------------------
# Launch Gradio interface
# -----------------------------
iface = gr.Interface(
    fn=answer_query,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question..."),
    outputs="text",
    title="RAG Chatbot",
    description="Ask questions about your documents. Answers are based only on provided context.",
)

if __name__ == "__main__":
    iface.launch(share=True, pwa=True)
