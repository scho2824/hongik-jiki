from setuptools import setup, find_packages

setup(
    name="hongikjiki",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.0.267",
        "openai>=1.0.0",
        "sentence-transformers>=2.2.2",
        "chromadb>=0.4.13",
        "flask>=2.3.2",
        "python-dotenv>=1.0.0",
        "striprtf>=0.0.25",
        "python-docx>=0.8.11",
        "PyPDF2>=3.0.1",
    ],
    entry_points={
        "console_scripts": [
            "hongikjiki-cli=hongikjiki.cli:main",
            "hongikjiki-web=hongikjiki.web:main",
        ],
    },
)