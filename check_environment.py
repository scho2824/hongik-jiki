# check_environment.py
import importlib
import sys

required_modules = [
    "openai", "chromadb", "langchain", "sentence_transformers", 
    "dotenv", "gradio", "flask"
]

missing = []
for module in required_modules:
    try:
        importlib.import_module(module)
        print(f"✅ {module} is installed")
    except ImportError:
        missing.append(module)
        print(f"❌ {module} is NOT installed")

if missing:
    print("\nInstall missing modules with:")
    print(f"pip install {' '.join(missing)}")
else:
    print("\nAll required modules are installed! 🎉")