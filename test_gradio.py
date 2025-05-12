#!/usr/bin/env python3
"""
Minimal script to test Gradio functionality
"""
import os
import sys
import gradio as gr

def greet(name):
    return f"Hello, {name}!"

# Create a simple interface
demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text",
    title="Test Gradio App",
)

if __name__ == "__main__":
    print("Starting Gradio test app...")
    try:
        # Launch with a short timeout to just verify it works
        demo.launch(prevent_thread_lock=False, share=False)
        print("Gradio launched successfully!")
    except Exception as e:
        print(f"Error launching Gradio: {e}")
        sys.exit(1)