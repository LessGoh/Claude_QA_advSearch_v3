"""
Hybrid Search Engine combining BM25 and Vector Search

This module implements a hybrid search system that combines keyword-based BM25 search
with vector similarity search using Reciprocal Rank Fusion (RRF) for optimal retrieval.
"""

import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple, Any, Optional
import logging
from dataclasses import dataclass
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result with score and metadata."""
    chunk_id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    rank: int


class HybridSearchEngine:
    """
    Hybrid search engine that combines BM25 keyword search with vector similarity search.
    """
    
    def __init__(self, rrf_k: int = 60):
        """
        Initialize hybrid search engine.
        
        Args:
            rrf_k: Parameter for Reciprocal Rank Fusion (default: 60)
        """
        self.rrf_k = rrf_k
        self.bm25_index = None
        self.documents = []  # List of document texts
        self.document_embeddings = []  # List of embeddings
        self.chunk_metadata = {}  # chunk_id -> metadata
        self.chunk_id_to_index = {}  # chunk_id -> index in documents list
        self.index_to_chunk_id = {}  # index -> chunk_id
        
        # BM25 parameters
        self.bm25_k1 = 1.5
        self.bm25_b = 0.75
        
        logger.info("Initialized HybridSearchEngine")
    
    def build_index(self, chunks: List[Dict[str, Any]], embeddings: List[np.ndarray]) -> None:
        """
        Build search index from document chunks and embeddings.
        
        Args:
            chunks: List of document chunks with metadata
            embeddings: List of embedding vectors corresponding to chunks
        """
        logger.info(f"Building hybrid search index with {len(chunks)} chunks")
        
        # Reset indices
        self.documents = []
        self.document_embeddings = embeddings
        self.chunk_metadata = {}
        self.chunk_id_to_index = {}
        self.index_to_chunk_id = {}
        
        # Process chunks
        for i, chunk in enumerate(chunks):
            chunk_id = chunk['chunk_id']
            content = chunk['content']
            
            # Store document text and metadata
            self.documents.append(content)
            self.chunk_metadata[chunk_id] = chunk
            self.chunk_id_to_index[chunk_id] = i
            self.index_to_chunk_id[i] = chunk_id
        
        # Build BM25 index
        self._build_bm25_index()
        
        logger.info("Hybrid search index built successfully")
    
    def _build_bm25_index(self) -> None:
        """Build BM25 index from documents."""
        if not self.documents:
            logger.warning("No documents to index")
            return
        
        # Tokenize documents for BM25
        tokenized_docs = []
        for doc in self.documents:
            # Simple tokenization (can be enhanced with better preprocessing)
            tokens = self._tokenize_text(doc)
            tokenized_docs.append(tokens)
        
        # Create BM25 index
        self.bm25_index = BM25Okapi(tokenized_docs, k1=self.bm25_k1, b=self.bm25_b)
        
        logger.debug(f"BM25 index created with {len(tokenized_docs)} documents")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and split by whitespace
        tokens = re.findall(r'\b[a-z]+\b', text)
        
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def search(self, query: str, query_embedding: Optional[np.ndarray] = None,
              top_k: int = 10, alpha: float = 0.5, 
              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Args:
            query: Search query text
            query_embedding: Pre-computed query embedding (optional)
            top_k: Number of results to return
            alpha: Weight for BM25 vs vector search (0.0 = pure vector, 1.0 = pure BM25)
            filters: Optional filters for results
            
        Returns:
            List of search results ranked by hybrid score
        """
        if not self.documents:
            logger.warning("No documents indexed")
            return []
        
        # Get BM25 results
        bm25_results = self._bm25_search(query, top_k * 2)  # Get more for fusion
        
        # Get vector search results if embedding provided
        vector_results = []
        if query_embedding is not None:
            vector_results = self._vector_search(query_embedding, top_k * 2)
        
        # Combine results using Reciprocal Rank Fusion
        if vector_results and alpha > 0:
            hybrid_results = self._reciprocal_rank_fusion(bm25_results, vector_results, alpha)
        else:
            # Pure BM25 search
            hybrid_results = bm25_results
        
        # Apply filters if provided
        if filters:
            hybrid_results = self._apply_filters(hybrid_results, filters)
        
        # Return top k results
        return hybrid_results[:top_k]
    
    def _bm25_search(self, query: str, top_k: int) -> List[SearchResult]:
        """
        Perform BM25 keyword search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        if self.bm25_index is None:
            logger.warning("BM25 index not built")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize_text(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Create search results
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:  # Only include positive scores
                chunk_id = self.index_to_chunk_id[idx]
                result = SearchResult(
                    chunk_id=chunk_id,
                    score=float(scores[idx]),
                    content=self.documents[idx],
                    metadata=self.chunk_metadata[chunk_id],
                    rank=rank + 1
                )
                results.append(result)
        
        return results
    
    def _vector_search(self, query_embedding: np.ndarray, top_k: int) -> List[SearchResult]:
        """
        Perform vector similarity search.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        if not self.document_embeddings:
            logger.warning("No document embeddings available")
            return []
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Create search results
        results = []
        for rank, idx in enumerate(top_indices):
            if similarities[idx] > 0:  # Only include positive similarities
                chunk_id = self.index_to_chunk_id[idx]
                result = SearchResult(
                    chunk_id=chunk_id,
                    score=float(similarities[idx]),
                    content=self.documents[idx],
                    metadata=self.chunk_metadata[chunk_id],
                    rank=rank + 1
                )
                results.append(result)
        
        return results
    
    def _reciprocal_rank_fusion(self, bm25_results: List[SearchResult], 
                               vector_results: List[SearchResult],
                               alpha: float = 0.5) -> List[SearchResult]:
        """
        Combine search results using Reciprocal Rank Fusion.
        
        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search
            alpha: Weight for BM25 vs vector search
            
        Returns:
            Fused search results
        """
        # Initialize RRF scores
        rrf_scores = {}
        
        # Add BM25 results to RRF
        for result in bm25_results:
            chunk_id = result.chunk_id
            rrf_score = alpha * (1 / (self.rrf_k + result.rank))
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score
        
        # Add vector search results to RRF
        for result in vector_results:
            chunk_id = result.chunk_id
            rrf_score = (1 - alpha) * (1 / (self.rrf_k + result.rank))
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score
        
        # Sort by RRF score
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Create fused results
        fused_results = []
        for rank, chunk_id in enumerate(sorted_chunk_ids):
            idx = self.chunk_id_to_index[chunk_id]
            result = SearchResult(
                chunk_id=chunk_id,
                score=rrf_scores[chunk_id],
                content=self.documents[idx],
                metadata=self.chunk_metadata[chunk_id],
                rank=rank + 1
            )
            fused_results.append(result)
        
        return fused_results
    
    def _apply_filters(self, results: List[SearchResult], 
                      filters: Dict[str, Any]) -> List[SearchResult]:
        """
        Apply filters to search results.
        
        Args:
            results: Search results to filter
            filters: Filter criteria
            
        Returns:
            Filtered search results
        """
        filtered_results = []
        
        for result in results:
            metadata = result.metadata
            include_result = True
            
            # Apply each filter
            for filter_key, filter_value in filters.items():
                if filter_key in metadata:
                    if isinstance(filter_value, list):
                        # Multiple allowed values
                        if metadata[filter_key] not in filter_value:
                            include_result = False
                            break
                    else:
                        # Single value match
                        if metadata[filter_key] != filter_value:
                            include_result = False
                            break
            
            if include_result:
                filtered_results.append(result)
        
        return filtered_results
    
    def search_by_section(self, query: str, section_id: str, 
                         query_embedding: Optional[np.ndarray] = None,
                         top_k: int = 10) -> List[SearchResult]:
        """
        Search within a specific document section.
        
        Args:
            query: Search query
            section_id: Section identifier to search within
            query_embedding: Pre-computed query embedding
            top_k: Number of results to return
            
        Returns:
            Search results filtered by section
        """
        filters = {'section_id': section_id}
        return self.search(query, query_embedding, top_k, filters=filters)
    
    def search_by_content_type(self, query: str, content_type: str,
                              query_embedding: Optional[np.ndarray] = None,
                              top_k: int = 10) -> List[SearchResult]:
        """
        Search for specific content types (e.g., formulas, tables).
        
        Args:
            query: Search query
            content_type: Type of content to search for
            query_embedding: Pre-computed query embedding
            top_k: Number of results to return
            
        Returns:
            Search results filtered by content type
        """
        filters = {'content_type': content_type}
        return self.search(query, query_embedding, top_k, filters=filters)
    
    def get_similar_chunks(self, chunk_id: str, top_k: int = 5) -> List[SearchResult]:
        """
        Find chunks similar to a given chunk.
        
        Args:
            chunk_id: Reference chunk ID
            top_k: Number of similar chunks to return
            
        Returns:
            List of similar chunks
        """
        if chunk_id not in self.chunk_id_to_index:
            logger.warning(f"Chunk {chunk_id} not found in index")
            return []
        
        # Get the embedding for the reference chunk
        ref_idx = self.chunk_id_to_index[chunk_id]
        if ref_idx >= len(self.document_embeddings):
            logger.warning(f"No embedding found for chunk {chunk_id}")
            return []
        
        ref_embedding = self.document_embeddings[ref_idx]
        
        # Perform vector search
        results = self._vector_search(ref_embedding, top_k + 1)  # +1 to exclude self
        
        # Remove the reference chunk from results
        filtered_results = [r for r in results if r.chunk_id != chunk_id]
        
        return filtered_results[:top_k]
    
    def save_index(self, filepath: str) -> None:
        """
        Save the search index to disk.
        
        Args:
            filepath: Path to save the index
        """
        index_data = {
            'documents': self.documents,
            'document_embeddings': self.document_embeddings,
            'chunk_metadata': self.chunk_metadata,
            'chunk_id_to_index': self.chunk_id_to_index,
            'index_to_chunk_id': self.index_to_chunk_id,
            'rrf_k': self.rrf_k
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        
        # Save BM25 index separately
        if self.bm25_index:
            bm25_filepath = filepath.replace('.pkl', '_bm25.pkl')
            with open(bm25_filepath, 'wb') as f:
                pickle.dump(self.bm25_index, f)
        
        logger.info(f"Search index saved to {filepath}")
    
    def load_index(self, filepath: str) -> None:
        """
        Load search index from disk.
        
        Args:
            filepath: Path to load the index from
        """
        try:
            with open(filepath, 'rb') as f:
                index_data = pickle.load(f)
            
            self.documents = index_data['documents']
            self.document_embeddings = index_data['document_embeddings']
            self.chunk_metadata = index_data['chunk_metadata']
            self.chunk_id_to_index = index_data['chunk_id_to_index']
            self.index_to_chunk_id = index_data['index_to_chunk_id']
            self.rrf_k = index_data.get('rrf_k', 60)
            
            # Load BM25 index
            bm25_filepath = filepath.replace('.pkl', '_bm25.pkl')
            if os.path.exists(bm25_filepath):
                with open(bm25_filepath, 'rb') as f:
                    self.bm25_index = pickle.load(f)
            else:
                # Rebuild BM25 index if not found
                self._build_bm25_index()
            
            logger.info(f"Search index loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading search index: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the search index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'total_documents': len(self.documents),
            'total_embeddings': len(self.document_embeddings),
            'has_bm25_index': self.bm25_index is not None,
            'rrf_k_parameter': self.rrf_k,
            'content_types': list(set(
                metadata.get('content_type', 'unknown') 
                for metadata in self.chunk_metadata.values()
            )),
            'sections': list(set(
                metadata.get('section_id', 'unknown')
                for metadata in self.chunk_metadata.values()
                if metadata.get('section_id')
            ))
        }