import gradio as gr

gr.load_chat("http://localhost:11434/v1/", model="llama3", token="***").launch()