# diagnose_loader.py
import os
import logging
from dotenv import load_dotenv
from hongikjiki.text_processing.document_loader import DocumentLoader
from hongikjiki.text_processing.text_normalizer import TextNormalizer

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DiagnosticTool")

# Load environment variables
load_dotenv()

def diagnose_document_loading():
    """Test document loading and normalization process"""
    print("\n=== Document Loading Diagnostics ===")
    
    # Initialize components
    loader = DocumentLoader()
    normalizer = TextNormalizer()
    
    # Check data directory
    data_dir = "data/jungbub_teachings"
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory {data_dir} does not exist!")
        return
    
    print(f"Data directory exists: {data_dir}")
    
    # Scan directory structure
    print("\nDirectory structure:")
    file_count = 0
    for root, dirs, files in os.walk(data_dir):
        rel_path = os.path.relpath(root, "data")
        print(f"- {rel_path}/")
        for file in files:
            if file.endswith(('.txt', '.rtf', '.pdf', '.docx')):
                file_count += 1
                if file_count <= 5:  # Only show first 5 files
                    print(f"  - {file}")
    
    print(f"\nTotal files found: {file_count}")
    
    # Test document loading with first file found
    test_file = None
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.txt', '.rtf', '.pdf', '.docx')):
                test_file = os.path.join(root, file)
                break
        if test_file:
            break
    
    if not test_file:
        print("No suitable test files found!")
        return
    
    print(f"\nTesting document loading with: {test_file}")
    
    try:
        # Test document loading
        doc_data = loader.load_document(test_file)
        
        if not doc_data:
            print(f"ERROR: Document loading returned None for {test_file}")
            return
        
        print(f"Document loaded successfully: {len(doc_data['content'])} characters")
        print(f"Metadata: {doc_data['metadata']}")
        
        # Test text normalization
        try:
            normalized_text = normalizer.normalize(doc_data['content'])
            print(f"Text normalized successfully: {len(normalized_text)} characters")
            print("\nSample normalized text (first 200 chars):")
            print(normalized_text[:200] + "...")
        except Exception as e:
            print(f"ERROR during text normalization: {e}")
            import traceback
            print(traceback.format_exc())
    
    except Exception as e:
        print(f"ERROR during document loading: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    diagnose_document_loading()