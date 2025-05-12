import os
import gradio as gr
import tempfile
import time
import json
from hongikjiki.utils import load_dotenv
from hongikjiki.langchain_integration.llm import get_llm
from hongikjiki.vector_store.embeddings import get_embeddings

# Load environment variables
load_dotenv()

# Verify API key is available
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in .env file.")
print(f"API key loaded: {api_key[:5]}...{api_key[-5:]}")

# Create a simple demo interface 
def greet(name):
    return f"Hello, {name}! This is a basic Gradio app to confirm the server is running."

demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text",
    title="Hongik-Jiki Demo App",
    description="This is a simple demo app to verify Gradio is working."
)

if __name__ == "__main__":
    demo.launch(share=True)  # Add share=True to create a public link