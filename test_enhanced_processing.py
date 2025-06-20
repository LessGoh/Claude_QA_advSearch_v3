#!/usr/bin/env python3
"""
Test script for enhanced document processing v3
Tests the system with sample financial PDF documents
"""

import os
import sys
from pathlib import Path

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from utils.enhanced_document_processor import EnhancedDocumentProcessor
from utils.config import get_api_keys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_processing():
    """Test enhanced processing with sample documents"""
    
    print("🚀 Testing Enhanced Document Processing v3")
    print("=" * 50)
    
    # Get API keys
    openai_key, pinecone_key, using_secrets = get_api_keys()
    
    if not openai_key or not pinecone_key:
        print("❌ API keys not configured")
        print("Please set OPENAI_API_KEY and PINECONE_API_KEY environment variables")
        return False
    
    print(f"✅ API keys loaded (using secrets: {using_secrets})")
    
    # Initialize enhanced processor
    try:
        processor = EnhancedDocumentProcessor(openai_key, pinecone_key)
        print("✅ Enhanced processor initialized")
    except Exception as e:
        print(f"❌ Error initializing processor: {e}")
        return False
    
    # Find sample documents
    sample_dir = Path("/mnt/c/Users/lesgoh/Desktop/QAbot/New_App_V1_Final/Example PDF")
    
    if not sample_dir.exists():
        print(f"❌ Sample directory not found: {sample_dir}")
        return False
    
    pdf_files = list(sample_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {sample_dir}")
        return False
    
    print(f"📁 Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"  • {pdf_file.name}")
    
    # Test with first document
    test_pdf = pdf_files[0]
    print(f"\n🔬 Testing with: {test_pdf.name}")
    print("-" * 50)
    
    try:
        # Process document
        result = processor.process_document(
            str(test_pdf),
            "pdf-qa-test-v3",  # Test index
            document_id=None
        )
        
        if result['status'] == 'success':
            print("✅ Document processing successful!")
            print(f"📄 Document Title: {result['document_title']}")
            print(f"📊 Chunks Created: {result['chunks_created']}")
            print(f"📝 Sections Detected: {result['sections_detected']}")
            print(f"🧮 Has Formulas: {result['has_formulas']}")
            print(f"📋 Has Tables: {result['has_tables']}")
            
            # Processing stats
            stats = result['processing_stats']
            print(f"\n📈 Processing Statistics:")
            print(f"  • Total Pages: {stats['total_pages']}")
            print(f"  • Total Chunks: {stats['total_chunks']}")
            print(f"  • Average Chunk Size: {stats['avg_chunk_size']:.0f} chars")
            print(f"  • Formula Chunks: {stats['formula_chunks']}")
            print(f"  • Table Chunks: {stats['table_chunks']}")
            
        else:
            print(f"❌ Document processing failed: {result.get('error_message', 'Unknown error')}")
            return False
        
        # Test query processing
        print(f"\n💬 Testing Query Processing")
        print("-" * 30)
        
        test_questions = [
            "What is the main research question?",
            "What methodology is used?",
            "What are the key findings?"
        ]
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            
            try:
                query_result = processor.query_documents(
                    question=question,
                    index_name="pdf-qa-test-v3",
                    top_k=3
                )
                
                if query_result['status'] == 'success':
                    print("✅ Query processed successfully")
                    
                    # Show search stats
                    search_stats = query_result['search_stats']
                    print(f"🔍 Retrieved {search_stats['chunks_retrieved']} chunks")
                    print(f"📊 Average score: {search_stats['avg_search_score']:.3f}")
                    
                    # Show formatted response (first 200 chars)
                    response = query_result['formatted_response']
                    preview = response[:200] + "..." if len(response) > 200 else response
                    print(f"💭 Response preview: {preview}")
                    
                else:
                    print(f"❌ Query failed: {query_result.get('error_message', 'Unknown error')}")
                
            except Exception as e:
                print(f"❌ Query error: {e}")
                
        # System statistics
        print(f"\n📊 System Statistics")
        print("-" * 20)
        stats = processor.get_system_stats()
        print(f"📚 Citations indexed: {stats['citations_count']}")
        print(f"📖 Documents indexed: {stats['documents_indexed']}")
        
        processors_status = stats['processors_initialized']
        print(f"🔧 Components initialized:")
        for component, status in processors_status.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {component}")
        
        print(f"\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.exception("Test error details:")
        return False

def test_individual_components():
    """Test individual components separately"""
    print(f"\n🧪 Testing Individual Components")
    print("=" * 40)
    
    # Test PDF processor
    try:
        from processors.pdf_processor import EnhancedPDFProcessor
        pdf_processor = EnhancedPDFProcessor()
        print("✅ EnhancedPDFProcessor initialized")
    except Exception as e:
        print(f"❌ EnhancedPDFProcessor error: {e}")
    
    # Test metadata extractor
    try:
        from processors.metadata_extractor import MetadataExtractor
        metadata_extractor = MetadataExtractor()
        print("✅ MetadataExtractor initialized")
    except Exception as e:
        print(f"❌ MetadataExtractor error: {e}")
    
    # Test smart chunker
    try:
        from utils.smart_chunker import SmartChunker
        smart_chunker = SmartChunker()
        print("✅ SmartChunker initialized")
    except Exception as e:
        print(f"❌ SmartChunker error: {e}")
    
    # Test citation manager
    try:
        from utils.citation_manager import CitationManager
        citation_manager = CitationManager()
        print("✅ CitationManager initialized")
    except Exception as e:
        print(f"❌ CitationManager error: {e}")
    
    # Test hybrid search
    try:
        from models.hybrid_search import HybridSearchEngine
        search_engine = HybridSearchEngine()
        print("✅ HybridSearchEngine initialized")
    except Exception as e:
        print(f"❌ HybridSearchEngine error: {e}")
    
    # Test response formatter
    try:
        from utils.response_formatter import ResponseFormatter
        formatter = ResponseFormatter()
        print("✅ ResponseFormatter initialized")
    except Exception as e:
        print(f"❌ ResponseFormatter error: {e}")

if __name__ == "__main__":
    print("🧪 Enhanced Document Processing v3 Test Suite")
    print("=" * 60)
    
    # Test individual components first
    test_individual_components()
    
    # Test full system integration
    success = test_enhanced_processing()
    
    if success:
        print("\n🎉 All tests passed! System is ready for use.")
    else:
        print("\n⚠️  Some tests failed. Please check the logs above.")
    
    print("\n" + "=" * 60)