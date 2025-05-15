# check_imports.py
import sys
from pathlib import Path

# Add project root to path if needed
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

print("Checking imports...")
try:
    from hongikjiki.modules.text_processing.document_loader import DocumentLoader
    print("✅ Successfully imported DocumentLoader")
except ImportError as e:
    print(f"❌ Error importing DocumentLoader: {e}")

try:
    from hongikjiki.modules.text_processing.document_processor import DocumentProcessor
    print("✅ Successfully imported DocumentProcessor")
except ImportError as e:
    print(f"❌ Error importing DocumentProcessor: {e}")

try:
    from hongikjiki.modules.text_processing.text_normalizer import TextNormalizer
    print("✅ Successfully imported TextNormalizer")
except ImportError as e:
    print(f"❌ Error importing TextNormalizer: {e}")

try:
    from hongikjiki.modules.text_processing.document_chunker import DocumentChunker
    print("✅ Successfully imported DocumentChunker")
except ImportError as e:
    print(f"❌ Error importing DocumentChunker: {e}")