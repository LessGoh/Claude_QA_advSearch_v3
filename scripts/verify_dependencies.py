# Verify all dependencies are correctly installed
import importlib
import sys

dependencies = [
    "pdfplumber",
    "fitz",  # pymupdf module
    "rank_bm25",
    "spacy"
]

def verify_dependencies():
    all_installed = True
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            print(f"✓ {dep} successfully imported")
        except ImportError:
            all_installed = False
            print(f"✗ Failed to import {dep}")
    
    # Verify spaCy model
    try:
        import spacy
        nlp = spacy.load("en_core_web_lg")
        print("✓ en_core_web_lg model successfully loaded")
    except:
        all_installed = False
        print("✗ Failed to load en_core_web_lg model")
    
    return all_installed

if __name__ == "__main__":
    success = verify_dependencies()
    sys.exit(0 if success else 1)