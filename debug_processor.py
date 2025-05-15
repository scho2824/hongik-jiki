# debug_processor.py
from pathlib import Path
import logging
from hongikjiki.modules.text_processing.document_processor import DocumentProcessor
from hongikjiki.modules.text_processing.document_loader import DocumentLoader

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("DebugProcessor")

# Define paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "jungbub_teachings"

# First, test just the document loader with a specific file
loader = DocumentLoader()
processor = DocumentProcessor()

# List files in directory
print(f"Files in {DATA_DIR}:")
file_found = False
for file_path in DATA_DIR.glob("**/*"):
    if file_path.is_file():
        ext = file_path.suffix.lower()
        if ext in ['.txt', '.pdf', '.docx', '.rtf', '.md']:
            file_found = True
            print(f"Found file: {file_path}")
            
            # Try to load a single file
            print(f"Attempting to load file: {file_path}")
            doc = loader.load_document(str(file_path))
            
            if doc:
                print(f"✅ Successfully loaded: {file_path.name}")
                print(f"Content length: {len(doc.get('content', ''))}")
                
                # Try processing the document
                print(f"Processing document...")
                chunks = processor.process_file(file_path)
                print(f"Generated {len(chunks)} chunks")
                
                # Show first chunk sample
                if chunks:
                    print("Sample chunk content:")
                    print(chunks[0]['content'][:200] + "...")
                break  # Just test one file for now
            else:
                print(f"❌ Failed to load: {file_path.name}")

if not file_found:
    print("No supported files found in directory!")