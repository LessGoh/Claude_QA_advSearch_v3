"""
Citation Manager for Precise Source References

This module manages citations and provides precise source references with page numbers,
section information, and exact text coordinates for academic financial documents.
"""

import re
import fitz  # pymupdf
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CitationData:
    """Data structure for citation information."""
    chunk_id: str
    document_id: str
    document_title: str
    page_number: int
    section_id: Optional[str]
    section_title: Optional[str]
    text_snippet: str
    coordinates: Dict[str, float]  # {'x1': float, 'y1': float, 'x2': float, 'y2': float}
    context_before: str = ""
    context_after: str = ""


class CitationManager:
    """
    Manages citation data and provides precise source references for document chunks.
    """
    
    def __init__(self):
        self.citations_index: Dict[str, CitationData] = {}
        self.document_citations: Dict[str, List[str]] = {}  # document_id -> chunk_ids
        
    def add_citation(self, chunk_data: Dict[str, Any], document_info: Dict[str, Any],
                    coordinates: Dict[str, float]) -> None:
        """
        Add citation information for a document chunk.
        
        Args:
            chunk_data: Chunk information from SmartChunker
            document_info: Document metadata and structure
            coordinates: Precise coordinates of the chunk in the PDF
        """
        chunk_id = chunk_data['chunk_id']
        document_id = document_info.get('document_id', '')
        
        # Find section information
        section_info = self._find_section_for_chunk(chunk_data, document_info)
        
        # Create citation data
        citation = CitationData(
            chunk_id=chunk_id,
            document_id=document_id,
            document_title=document_info.get('title', 'Unknown Document'),
            page_number=chunk_data.get('page_num', 1),
            section_id=section_info.get('section_id') if section_info else None,
            section_title=section_info.get('title') if section_info else None,
            text_snippet=self._create_text_snippet(chunk_data['content']),
            coordinates=coordinates
        )
        
        # Store citation
        self.citations_index[chunk_id] = citation
        
        # Update document index
        if document_id not in self.document_citations:
            self.document_citations[document_id] = []
        self.document_citations[document_id].append(chunk_id)
        
        logger.debug(f"Added citation for chunk {chunk_id}")
    
    def get_citation(self, chunk_id: str) -> Optional[CitationData]:
        """
        Retrieve citation data for a chunk.
        
        Args:
            chunk_id: Unique identifier for the chunk
            
        Returns:
            CitationData object or None if not found
        """
        return self.citations_index.get(chunk_id)
    
    def format_citation(self, chunk_id: str, include_quote: bool = True) -> Optional[str]:
        """
        Format citation string for a chunk.
        
        Args:
            chunk_id: Unique identifier for the chunk
            include_quote: Whether to include text quote in citation
            
        Returns:
            Formatted citation string or None if not found
        """
        citation = self.get_citation(chunk_id)
        if not citation:
            return None
        
        # Build basic citation
        citation_parts = [citation.document_title]
        
        if citation.page_number:
            citation_parts.append(f"p. {citation.page_number}")
        
        if citation.section_title:
            citation_parts.append(citation.section_title)
        
        formatted_citation = ", ".join(citation_parts)
        
        # Add quote if requested
        if include_quote and citation.text_snippet:
            formatted_citation += f' - "{citation.text_snippet}"'
        
        return formatted_citation
    
    def get_context_around_citation(self, chunk_id: str, pdf_path: str, 
                                   context_chars: int = 200) -> Optional[Dict[str, str]]:
        """
        Retrieve context around a citation by re-reading the PDF.
        
        Args:
            chunk_id: Unique identifier for the chunk
            pdf_path: Path to the original PDF file
            context_chars: Number of characters to include before and after
            
        Returns:
            Dictionary with 'before', 'chunk', 'after' text or None if not found
        """
        citation = self.get_citation(chunk_id)
        if not citation:
            return None
        
        try:
            # Re-read the PDF area using stored coordinates
            doc = fitz.open(pdf_path)
            page = doc[citation.page_number - 1]  # Convert to 0-based indexing
            
            # Extract text from the exact coordinates with expanded area for context
            coords = citation.coordinates
            
            # Expand coordinates for context
            expanded_rect = fitz.Rect(
                max(0, coords['x1'] - 50),
                max(0, coords['y1'] - 50),
                min(page.rect.width, coords['x2'] + 50),
                min(page.rect.height, coords['y2'] + 50)
            )
            
            # Get text from expanded area
            full_text = page.get_text("text", clip=expanded_rect)
            
            # Find the chunk text within the full text
            chunk_text = citation.text_snippet
            chunk_start = full_text.find(chunk_text[:50])  # Use first 50 chars to find position
            
            if chunk_start == -1:
                # Fallback: return available text
                doc.close()
                return {
                    'before': '',
                    'chunk': citation.text_snippet,
                    'after': '',
                    'full_context': full_text
                }
            
            # Extract context
            context_start = max(0, chunk_start - context_chars)
            context_end = min(len(full_text), chunk_start + len(chunk_text) + context_chars)
            
            before_text = full_text[context_start:chunk_start]
            after_text = full_text[chunk_start + len(chunk_text):context_end]
            
            doc.close()
            
            return {
                'before': before_text,
                'chunk': chunk_text,
                'after': after_text,
                'full_context': full_text[context_start:context_end]
            }
            
        except Exception as e:
            logger.error(f"Error retrieving context for chunk {chunk_id}: {e}")
            return None
    
    def get_citations_for_document(self, document_id: str) -> List[CitationData]:
        """
        Get all citations for a specific document.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            List of CitationData objects
        """
        chunk_ids = self.document_citations.get(document_id, [])
        return [self.citations_index[chunk_id] for chunk_id in chunk_ids 
                if chunk_id in self.citations_index]
    
    def find_citations_by_section(self, document_id: str, section_id: str) -> List[CitationData]:
        """
        Find citations within a specific document section.
        
        Args:
            document_id: Unique identifier for the document
            section_id: Section identifier
            
        Returns:
            List of CitationData objects in the section
        """
        document_citations = self.get_citations_for_document(document_id)
        return [citation for citation in document_citations 
                if citation.section_id == section_id]
    
    def find_citations_by_page(self, document_id: str, page_number: int) -> List[CitationData]:
        """
        Find citations on a specific page.
        
        Args:
            document_id: Unique identifier for the document
            page_number: Page number (1-based)
            
        Returns:
            List of CitationData objects on the page
        """
        document_citations = self.get_citations_for_document(document_id)
        return [citation for citation in document_citations 
                if citation.page_number == page_number]
    
    def create_detailed_citation(self, chunk_ids: List[str], 
                               response_text: str = "") -> Dict[str, Any]:
        """
        Create a detailed citation with multiple sources.
        
        Args:
            chunk_ids: List of chunk IDs that contributed to the response
            response_text: The generated response text
            
        Returns:
            Dictionary with detailed citation information
        """
        citations = []
        sources_by_document = {}
        
        for chunk_id in chunk_ids:
            citation = self.get_citation(chunk_id)
            if citation:
                citations.append(citation)
                
                doc_id = citation.document_id
                if doc_id not in sources_by_document:
                    sources_by_document[doc_id] = {
                        'document_title': citation.document_title,
                        'citations': []
                    }
                sources_by_document[doc_id]['citations'].append(citation)
        
        # Create formatted citation list
        formatted_sources = []
        for doc_id, doc_info in sources_by_document.items():
            doc_citations = doc_info['citations']
            
            # Group by page for cleaner citations
            pages = {}
            for citation in doc_citations:
                page = citation.page_number
                if page not in pages:
                    pages[page] = []
                pages[page].append(citation)
            
            # Format citations by document
            page_refs = []
            for page, page_citations in sorted(pages.items()):
                sections = list(set([c.section_title for c in page_citations if c.section_title]))
                if sections:
                    page_refs.append(f"p. {page} ({', '.join(sections)})")
                else:
                    page_refs.append(f"p. {page}")
            
            formatted_source = {
                'document_title': doc_info['document_title'],
                'page_references': page_refs,
                'citation_string': f"{doc_info['document_title']}, {', '.join(page_refs)}",
                'quotes': [c.text_snippet for c in doc_citations if c.text_snippet]
            }
            formatted_sources.append(formatted_source)
        
        return {
            'sources': formatted_sources,
            'citation_count': len(citations),
            'document_count': len(sources_by_document),
            'raw_citations': citations
        }
    
    def export_citations(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Export citation data for backup or analysis.
        
        Args:
            document_id: If provided, export only citations for this document
            
        Returns:
            Dictionary with citation data
        """
        if document_id:
            citations = self.get_citations_for_document(document_id)
        else:
            citations = list(self.citations_index.values())
        
        # Convert to serializable format
        export_data = {
            'citations': [],
            'export_timestamp': str(fitz.time.time()),
            'total_citations': len(citations)
        }
        
        for citation in citations:
            export_data['citations'].append({
                'chunk_id': citation.chunk_id,
                'document_id': citation.document_id,
                'document_title': citation.document_title,
                'page_number': citation.page_number,
                'section_id': citation.section_id,
                'section_title': citation.section_title,
                'text_snippet': citation.text_snippet,
                'coordinates': citation.coordinates
            })
        
        return export_data
    
    def import_citations(self, citation_data: Dict[str, Any]) -> None:
        """
        Import citation data from backup.
        
        Args:
            citation_data: Citation data dictionary from export_citations()
        """
        for citation_dict in citation_data.get('citations', []):
            citation = CitationData(
                chunk_id=citation_dict['chunk_id'],
                document_id=citation_dict['document_id'],
                document_title=citation_dict['document_title'],
                page_number=citation_dict['page_number'],
                section_id=citation_dict.get('section_id'),
                section_title=citation_dict.get('section_title'),
                text_snippet=citation_dict['text_snippet'],
                coordinates=citation_dict['coordinates']
            )
            
            self.citations_index[citation.chunk_id] = citation
            
            # Update document index
            if citation.document_id not in self.document_citations:
                self.document_citations[citation.document_id] = []
            if citation.chunk_id not in self.document_citations[citation.document_id]:
                self.document_citations[citation.document_id].append(citation.chunk_id)
        
        logger.info(f"Imported {len(citation_data.get('citations', []))} citations")
    
    def _find_section_for_chunk(self, chunk_data: Dict[str, Any], 
                               document_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find the section that contains this chunk."""
        sections = document_info.get('sections', [])
        chunk_page = chunk_data.get('page_num', 1)
        chunk_section_id = chunk_data.get('section_id')
        
        # First try direct section ID match
        if chunk_section_id:
            for section in sections:
                if section.get('section_id') == chunk_section_id:
                    return section
        
        # Fallback: find section by page number
        best_section = None
        for section in sections:
            if section.get('page_num', 1) <= chunk_page:
                if not best_section or section.get('page_num', 1) > best_section.get('page_num', 1):
                    best_section = section
        
        return best_section
    
    def _create_text_snippet(self, content: str, max_length: int = 200) -> str:
        """Create a representative text snippet from chunk content."""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content.strip())
        
        # If content is short enough, return as-is
        if len(content) <= max_length:
            return content
        
        # Try to find a good breaking point (sentence end)
        truncated = content[:max_length]
        
        # Look for last sentence end
        sentence_end = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )
        
        if sentence_end > max_length * 0.6:  # If we found a good breaking point
            return truncated[:sentence_end + 1]
        else:
            # Fallback: break at word boundary
            word_end = truncated.rfind(' ')
            if word_end > 0:
                return truncated[:word_end] + "..."
            else:
                return truncated + "..."