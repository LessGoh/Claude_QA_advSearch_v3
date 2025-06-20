#!/usr/bin/env python3
"""
Simple test for individual components
Tests components that don't require external APIs
"""

import sys
from pathlib import Path

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_components():
    """Test individual components"""
    print("🧪 Testing Enhanced Components v3")
    print("=" * 40)
    
    results = {}
    
    # Test PDF processor
    print("1️⃣ Testing EnhancedPDFProcessor...")
    try:
        from processors.pdf_processor import EnhancedPDFProcessor
        pdf_processor = EnhancedPDFProcessor()
        print("   ✅ EnhancedPDFProcessor initialized")
        results['pdf_processor'] = True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['pdf_processor'] = False
    
    # Test metadata extractor
    print("2️⃣ Testing MetadataExtractor...")
    try:
        from processors.metadata_extractor import MetadataExtractor
        metadata_extractor = MetadataExtractor()
        print("   ✅ MetadataExtractor initialized")
        results['metadata_extractor'] = True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['metadata_extractor'] = False
    
    # Test smart chunker
    print("3️⃣ Testing SmartChunker...")
    try:
        from utils.smart_chunker import SmartChunker
        smart_chunker = SmartChunker()
        print("   ✅ SmartChunker initialized")
        results['smart_chunker'] = True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['smart_chunker'] = False
    
    # Test citation manager
    print("4️⃣ Testing CitationManager...")
    try:
        from utils.citation_manager import CitationManager
        citation_manager = CitationManager()
        print("   ✅ CitationManager initialized")
        results['citation_manager'] = True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['citation_manager'] = False
    
    # Test hybrid search
    print("5️⃣ Testing HybridSearchEngine...")
    try:
        from models.hybrid_search import HybridSearchEngine
        search_engine = HybridSearchEngine()
        print("   ✅ HybridSearchEngine initialized")
        results['hybrid_search'] = True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['hybrid_search'] = False
    
    # Test response formatter
    print("6️⃣ Testing ResponseFormatter...")
    try:
        from utils.response_formatter import ResponseFormatter
        formatter = ResponseFormatter()
        print("   ✅ ResponseFormatter initialized")
        results['response_formatter'] = True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['response_formatter'] = False
    
    # Test financial RAG chain (without LLM client)
    print("7️⃣ Testing FinancialRAGChain...")
    try:
        from models.financial_rag_chain import FinancialRAGChain
        rag_chain = FinancialRAGChain(llm_client=None)  # No client for testing
        print("   ✅ FinancialRAGChain initialized")
        results['financial_rag'] = True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['financial_rag'] = False
    
    # Summary
    print("\n📊 Test Results Summary:")
    print("-" * 30)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component}")
    
    print(f"\n🎯 Results: {passed_tests}/{total_tests} components working")
    
    if passed_tests == total_tests:
        print("🎉 All components passed basic initialization!")
        return True
    else:
        print(f"⚠️  {total_tests - passed_tests} components need attention")
        return False

def test_sample_pdf_processing():
    """Test PDF processing with a sample document"""
    print("\n📄 Testing PDF Processing with Sample Document")
    print("=" * 50)
    
    # Check for sample PDFs
    sample_dir = Path("/mnt/c/Users/lesgoh/Desktop/QAbot/New_App_V1_Final/Example PDF")
    
    if not sample_dir.exists():
        print(f"❌ Sample directory not found: {sample_dir}")
        return False
    
    pdf_files = list(sample_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {sample_dir}")
        return False
    
    print(f"📁 Found {len(pdf_files)} PDF files:")
    for i, pdf_file in enumerate(pdf_files[:3]):  # Show first 3
        print(f"  {i+1}. {pdf_file.name}")
    
    # Test with first document
    test_pdf = pdf_files[0]
    print(f"\n🔬 Testing structure detection with: {test_pdf.name}")
    
    try:
        from processors.pdf_processor import EnhancedPDFProcessor
        from processors.metadata_extractor import MetadataExtractor
        
        # Test PDF processor
        pdf_processor = EnhancedPDFProcessor()
        print("📖 Processing document structure...")
        
        doc_info = pdf_processor.process_document(str(test_pdf))
        
        print(f"✅ Document processed successfully!")
        print(f"   📄 Pages: {doc_info.get('page_count', 0)}")
        print(f"   📝 Sections detected: {len(doc_info.get('sections', []))}")
        
        # Show first few sections
        sections = doc_info.get('sections', [])
        if sections:
            print(f"   📋 First few sections:")
            for i, section in enumerate(sections[:3]):
                print(f"      {i+1}. {section.get('title', 'Unknown')} (page {section.get('page_num', '?')})")
        
        # Test metadata extraction
        print("\n📊 Extracting metadata...")
        metadata_extractor = MetadataExtractor()
        metadata = metadata_extractor.extract_metadata(str(test_pdf))
        
        print(f"✅ Metadata extracted!")
        print(f"   📑 Title: {metadata.get('title', 'Not detected')}")
        print(f"   👥 Authors: {metadata.get('authors', [])}")
        print(f"   🏷️  Keywords: {len(metadata.get('keywords', []))} found")
        print(f"   💰 Financial indicators: {metadata.get('financial_indicators', {}).get('has_financial_terms', False)}")
        
        # Test smart chunking
        print("\n🧩 Testing smart chunking...")
        from utils.smart_chunker import SmartChunker
        
        chunker = SmartChunker()
        chunks = chunker.chunk_document(doc_info)
        
        print(f"✅ Smart chunking completed!")
        print(f"   📦 Chunks created: {len(chunks)}")
        
        # Analyze chunks
        text_chunks = [c for c in chunks if c.get('content_type') == 'text']
        formula_chunks = [c for c in chunks if c.get('formula_count', 0) > 0]
        table_chunks = [c for c in chunks if c.get('table_count', 0) > 0]
        
        print(f"   📄 Text chunks: {len(text_chunks)}")
        print(f"   🧮 Formula chunks: {len(formula_chunks)}")
        print(f"   📋 Table chunks: {len(table_chunks)}")
        
        if chunks:
            avg_size = sum(len(c.get('content', '')) for c in chunks) / len(chunks)
            print(f"   📏 Average chunk size: {avg_size:.0f} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during PDF processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Enhanced Document Processing v3 - Component Tests")
    print("=" * 60)
    
    # Test basic component initialization
    components_ok = test_components()
    
    if components_ok:
        # Test PDF processing if components are working
        pdf_test_ok = test_sample_pdf_processing()
        
        if pdf_test_ok:
            print("\n🎉 All tests completed successfully!")
            print("💡 The system is ready for integration with Streamlit!")
        else:
            print("\n⚠️  PDF processing tests had issues")
    else:
        print("\n❌ Component initialization failed")
    
    print("\n" + "=" * 60)