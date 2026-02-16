from rag_chain import llm, retriever
import threading
import time

def thinking_dots():
    while not done:
        print(".", end="", flush=True)
        time.sleep(0.5)

while True:
    query = input("\nAsk something (or 'exit'): ")
    if query.lower() == "exit":
        break

    # Retrieve relevant document chunks
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

    # Show thinking dots while model is processing
    done = False
    t = threading.Thread(target=thinking_dots)
    t.start()

    # Blocking call (CPU will take 2–5 min per reply)
    response = llm.invoke(prompt)

    # Stop spinner
    done = True
    t.join()

    print("\nAnswer:\n", response)
