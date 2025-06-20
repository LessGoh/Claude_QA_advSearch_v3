"""
Enhanced Document Processor

Integrates all new v3 components for comprehensive document processing
with structure detection, smart chunking, and citation management.
"""

import os
import tempfile
import openai
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

# Import our new components
from processors.pdf_processor import EnhancedPDFProcessor
from processors.metadata_extractor import MetadataExtractor
from utils.smart_chunker import SmartChunker
from utils.citation_manager import CitationManager
from models.hybrid_search import HybridSearchEngine
from models.financial_rag_chain import FinancialRAGChain
from utils.response_formatter import ResponseFormatter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedDocumentProcessor:
    """
    Comprehensive document processor that integrates all v3 enhancements
    for academic financial document processing.
    """
    
    def __init__(self, openai_api_key: str, pinecone_api_key: str):
        """
        Initialize the enhanced document processor.
        
        Args:
            openai_api_key: OpenAI API key for embeddings and LLM
            pinecone_api_key: Pinecone API key for vector storage
        """
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        
        # Initialize OpenAI client
        openai.api_key = openai_api_key
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=openai_api_key
        )
        
        # Initialize processors
        self.pdf_processor = EnhancedPDFProcessor()
        self.metadata_extractor = MetadataExtractor()
        self.smart_chunker = SmartChunker()
        self.citation_manager = CitationManager()
        self.search_engine = HybridSearchEngine()
        self.rag_chain = FinancialRAGChain(llm_client=self.openai_client)
        self.response_formatter = ResponseFormatter()
        
        logger.info("Enhanced Document Processor initialized")
    
    def process_document(self, file_path: str, index_name: str, 
                        document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a PDF document with enhanced v3 capabilities.
        
        Args:
            file_path: Path to PDF file
            index_name: Pinecone index name to store vectors
            document_id: Optional document ID (generated if not provided)
            
        Returns:
            Processing result with metadata and statistics
        """
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Step 1: Enhanced PDF processing with structure detection
            logger.info("Step 1: Analyzing document structure...")
            document_info = self.pdf_processor.process_document(file_path)
            
            # Step 2: Extract comprehensive metadata
            logger.info("Step 2: Extracting metadata...")
            metadata = self.metadata_extractor.extract_metadata(file_path)
            document_info['extracted_metadata'] = metadata
            
            # Generate document ID if not provided
            if not document_id:
                document_id = self._generate_document_id(file_path, metadata)
            document_info['document_id'] = document_id
            
            # Step 3: Smart chunking with formula/table preservation
            logger.info("Step 3: Smart chunking...")
            chunks = self.smart_chunker.chunk_document(document_info)
            
            # Step 4: Generate embeddings for chunks
            logger.info("Step 4: Generating embeddings...")
            chunk_embeddings = self._generate_chunk_embeddings(chunks)
            
            # Step 5: Store in Pinecone with enhanced metadata
            logger.info("Step 5: Storing in vector database...")
            storage_result = self._store_chunks_in_pinecone(
                chunks, chunk_embeddings, index_name, document_info
            )
            
            # Step 6: Build citation index
            logger.info("Step 6: Building citation index...")
            self._build_citation_index(chunks, document_info)
            
            # Step 7: Update search index
            logger.info("Step 7: Updating search index...")
            self._update_search_index(chunks, chunk_embeddings)
            
            processing_result = {
                'status': 'success',
                'document_id': document_id,
                'document_title': metadata.get('title', 'Unknown'),
                'chunks_created': len(chunks),
                'sections_detected': len(document_info.get('sections', [])),
                'metadata': metadata,
                'storage_result': storage_result,
                'has_formulas': any(chunk.get('formula_count', 0) > 0 for chunk in chunks),
                'has_tables': any(chunk.get('table_count', 0) > 0 for chunk in chunks),
                'processing_stats': {
                    'total_pages': document_info.get('page_count', 0),
                    'total_chunks': len(chunks),
                    'avg_chunk_size': np.mean([len(chunk['content']) for chunk in chunks]) if chunks else 0,
                    'formula_chunks': sum(1 for chunk in chunks if chunk.get('formula_count', 0) > 0),
                    'table_chunks': sum(1 for chunk in chunks if chunk.get('table_count', 0) > 0)
                }
            }
            
            logger.info(f"Document processing completed successfully: {document_id}")
            return processing_result
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'document_id': document_id or 'unknown'
            }
    
    def query_documents(self, question: str, index_name: str, 
                       top_k: int = 5, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Query documents using enhanced RAG with hybrid search.
        
        Args:
            question: User question
            index_name: Pinecone index to search
            top_k: Number of chunks to retrieve
            filters: Optional filters for search
            
        Returns:
            Enhanced response with citations and confidence
        """
        logger.info(f"Processing query: {question[:100]}...")
        
        try:
            # Step 1: Generate query embedding
            query_embedding = self.embeddings.embed_query(question)
            
            # Step 2: Hybrid search (BM25 + vector)
            if hasattr(self.search_engine, 'documents') and self.search_engine.documents:
                # Use local hybrid search if available
                search_results = self.search_engine.search(
                    query=question,
                    query_embedding=np.array(query_embedding),
                    top_k=top_k,
                    filters=filters
                )
                
                # Convert search results to chunks format
                retrieved_chunks = []
                for result in search_results:
                    chunk_data = result.metadata
                    chunk_data['content'] = result.content
                    chunk_data['search_score'] = result.score
                    retrieved_chunks.append(chunk_data)
            else:
                # Fallback to Pinecone-only search
                retrieved_chunks = self._fallback_pinecone_search(
                    query_embedding, index_name, top_k, filters
                )
            
            # Step 3: Get citations for retrieved chunks
            citations = []
            for chunk in retrieved_chunks:
                chunk_id = chunk.get('chunk_id')
                if chunk_id:
                    citation = self.citation_manager.get_citation(chunk_id)
                    citations.append(citation)
                else:
                    citations.append(None)
            
            # Step 4: Generate enhanced response using Financial RAG
            rag_response = self.rag_chain.generate_answer(question, retrieved_chunks, citations)
            
            # Step 5: Format response
            formatted_response = self.response_formatter.format_response(
                answer=rag_response.answer,
                explanation=rag_response.explanation,
                sources=self._convert_citations_to_sources(citations),
                confidence=rag_response.confidence
            )
            
            return {
                'status': 'success',
                'formatted_response': formatted_response,
                'raw_response': rag_response,
                'retrieved_chunks': retrieved_chunks,
                'citations': citations,
                'search_stats': {
                    'chunks_retrieved': len(retrieved_chunks),
                    'avg_search_score': np.mean([c.get('search_score', 0) for c in retrieved_chunks]) if retrieved_chunks else 0,
                    'has_formulas': any(c.get('formula_count', 0) > 0 for c in retrieved_chunks),
                    'has_tables': any(c.get('table_count', 0) > 0 for c in retrieved_chunks)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'formatted_response': self.response_formatter.format_response(
                    answer="Извините, произошла ошибка при обработке вашего вопроса.",
                    explanation=f"Детали ошибки: {str(e)}",
                    sources=[],
                    confidence={'level': 'Very Low', 'explanation': 'Техническая ошибка'}
                )
            }
    
    def _generate_document_id(self, file_path: str, metadata: Dict[str, Any]) -> str:
        """Generate unique document ID."""
        import hashlib
        
        # Use file name and metadata for ID generation
        base_name = os.path.basename(file_path)
        title = metadata.get('title', base_name)
        authors = str(metadata.get('authors', []))
        
        id_string = f"{base_name}_{title}_{authors}"
        return hashlib.md5(id_string.encode()).hexdigest()[:16]
    
    def _generate_chunk_embeddings(self, chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Generate embeddings for document chunks."""
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings in batches to avoid rate limits
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        return [np.array(emb) for emb in all_embeddings]
    
    def _store_chunks_in_pinecone(self, chunks: List[Dict[str, Any]], 
                                 embeddings: List[np.ndarray],
                                 index_name: str, 
                                 document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Store chunks in Pinecone with enhanced metadata."""
        try:
            index = self.pc.Index(index_name)
            
            # Prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                metadata = {
                    'content': chunk['content'],
                    'chunk_id': chunk['chunk_id'],
                    'document_id': document_info['document_id'],
                    'document_title': document_info.get('extracted_metadata', {}).get('title', 'Unknown'),
                    'page_num': chunk.get('page_num', 1),
                    'section_id': chunk.get('section_id', ''),
                    'section_title': chunk.get('section_title', ''),
                    'content_type': chunk.get('content_type', 'text'),
                    'formula_count': chunk.get('formula_count', 0),
                    'table_count': chunk.get('table_count', 0),
                    'character_count': chunk.get('character_count', 0),
                    'word_count': chunk.get('word_count', 0),
                    'upload_user': document_info.get('upload_user', 'unknown'),
                    'upload_date': document_info.get('upload_date', ''),
                    'index_name': index_name
                }
                
                vectors.append({
                    'id': chunk['chunk_id'],
                    'values': embedding.tolist(),
                    'metadata': metadata
                })
            
            # Upsert in batches
            batch_size = 100
            total_upserted = 0
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch)
                total_upserted += len(batch)
            
            return {
                'status': 'success',
                'vectors_stored': total_upserted,
                'index_name': index_name
            }
            
        except Exception as e:
            logger.error(f"Error storing chunks in Pinecone: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e)
            }
    
    def _build_citation_index(self, chunks: List[Dict[str, Any]], 
                             document_info: Dict[str, Any]) -> None:
        """Build citation index for precise source references."""
        for chunk in chunks:
            # Create coordinates from chunk metadata
            coordinates = {
                'x1': 0,  # Would be populated from actual PDF coordinates
                'y1': 0,
                'x2': 100,
                'y2': 100
            }
            
            self.citation_manager.add_citation(chunk, document_info, coordinates)
    
    def _update_search_index(self, chunks: List[Dict[str, Any]], 
                           embeddings: List[np.ndarray]) -> None:
        """Update the hybrid search index."""
        # Only update if we have existing search index
        if hasattr(self.search_engine, 'documents'):
            # For now, we'll rebuild the index
            # In production, this could be optimized for incremental updates
            pass
    
    def _fallback_pinecone_search(self, query_embedding: List[float], 
                                 index_name: str, top_k: int, 
                                 filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Fallback search using only Pinecone."""
        try:
            index = self.pc.Index(index_name)
            
            # Prepare filter
            pinecone_filter = {}
            if filters:
                for key, value in filters.items():
                    pinecone_filter[key] = {"$eq": value}
            
            # Query Pinecone
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=pinecone_filter if pinecone_filter else None
            )
            
            # Convert to our format
            retrieved_chunks = []
            for match in results.matches:
                chunk_data = match.metadata
                chunk_data['search_score'] = match.score
                retrieved_chunks.append(chunk_data)
            
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Error in fallback Pinecone search: {str(e)}")
            return []
    
    def _convert_citations_to_sources(self, citations: List[Any]) -> List[Dict[str, Any]]:
        """Convert citation objects to source format for response formatter."""
        sources = []
        
        for citation in citations:
            if citation:
                source = {
                    'citation': self.citation_manager.format_citation(citation.chunk_id, include_quote=False),
                    'quote': citation.text_snippet,
                    'document_title': citation.document_title,
                    'page_number': citation.page_number,
                    'section_title': citation.section_title
                }
                sources.append(source)
        
        return sources
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            'citations_count': len(self.citation_manager.citations_index),
            'documents_indexed': len(self.citation_manager.document_citations),
            'search_index_stats': self.search_engine.get_index_stats() if hasattr(self.search_engine, 'get_index_stats') else {},
            'processors_initialized': {
                'pdf_processor': self.pdf_processor is not None,
                'metadata_extractor': self.metadata_extractor is not None,
                'smart_chunker': self.smart_chunker is not None,
                'citation_manager': self.citation_manager is not None,
                'search_engine': self.search_engine is not None,
                'rag_chain': self.rag_chain is not None,
                'response_formatter': self.response_formatter is not None
            }
        }