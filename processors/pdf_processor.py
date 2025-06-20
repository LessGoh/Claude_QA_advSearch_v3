"""
Enhanced PDF Processor for Document Structure Detection

This module provides advanced PDF processing capabilities using both pymupdf and pdfplumber
to detect document structure, sections, and extract metadata with precise positioning information.
"""

import re
import fitz  # pymupdf
import pdfplumber
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPDFProcessor:
    """
    Enhanced PDF processor that combines pymupdf and pdfplumber for comprehensive
    document structure detection and text extraction.
    """
    
    def __init__(self):
        # Section patterns for academic papers
        self.section_patterns = [
            r'^(?:\d+\.)\s*([A-Z][^\n]+)$',  # Numbered sections: "1. Introduction"
            r'^([A-Z][A-Z\s]+)$',  # All caps headers: "METHODOLOGY"
            r'^(?:\d+\.\d+)\s*([A-Z][^\n]+)$',  # Subsections: "2.1 Model Setup"
            r'^(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$',  # Title case: "Literature Review" 
        ]
        
        # Font size thresholds
        self.header_font_threshold = 1.2  # Headers should be 20% larger than body text
        self.min_font_size = 8
        
    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process PDF document and extract structure information.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing document structure and metadata
        """
        logger.info(f"Processing document: {pdf_path}")
        
        try:
            # Load PDF with both libraries
            fitz_doc = fitz.open(pdf_path)
            
            # Extract basic document info
            doc_info = {
                'path': pdf_path,
                'page_count': len(fitz_doc),
                'metadata': self._extract_basic_metadata(fitz_doc),
                'sections': [],
                'pages': []
            }
            
            # Analyze document structure
            with pdfplumber.open(pdf_path) as pdfplumber_doc:
                doc_info['sections'] = self._detect_document_structure(fitz_doc, pdfplumber_doc)
                doc_info['pages'] = self._extract_page_info(fitz_doc, pdfplumber_doc)
            
            fitz_doc.close()
            return doc_info
            
        except Exception as e:
            logger.error(f"Error processing document {pdf_path}: {str(e)}")
            raise
    
    def _extract_basic_metadata(self, fitz_doc: fitz.Document) -> Dict[str, Any]:
        """Extract basic metadata from PDF document."""
        metadata = fitz_doc.metadata
        return {
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'subject': metadata.get('subject', ''),
            'creator': metadata.get('creator', ''),
            'producer': metadata.get('producer', ''),
            'creation_date': metadata.get('creationDate', ''),
            'modification_date': metadata.get('modDate', '')
        }
    
    def _detect_document_structure(self, fitz_doc: fitz.Document, pdfplumber_doc) -> List[Dict[str, Any]]:
        """
        Detect document sections using hybrid approach with pymupdf and pdfplumber.
        
        Args:
            fitz_doc: PyMuPDF document object
            pdfplumber_doc: pdfplumber document object
            
        Returns:
            List of detected sections with metadata
        """
        sections = []
        body_font_size = self._estimate_body_font_size(fitz_doc)
        logger.info(f"Estimated body font size: {body_font_size}")
        
        for page_num in range(len(fitz_doc)):
            fitz_page = fitz_doc[page_num]
            pdfplumber_page = pdfplumber_doc.pages[page_num] if page_num < len(pdfplumber_doc.pages) else None
            
            # Get text blocks with style information from pymupdf
            blocks = fitz_page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        font_size = span["size"]
                        font_flags = span["flags"]
                        
                        # Check if this could be a header
                        if self._is_potential_header(text, font_size, font_flags, body_font_size):
                            # Verify with pdfplumber positioning if available
                            if pdfplumber_page:
                                y_coord = span["bbox"][1]  # y-coordinate from top
                                x_coord = span["bbox"][0]  # x-coordinate from left
                                
                                # Check if this matches our section patterns
                                section_match = self._match_section_patterns(text)
                                if section_match:
                                    section_info = {
                                        'section_id': section_match['id'],
                                        'title': section_match['title'],
                                        'page_num': page_num + 1,
                                        'coordinates': {
                                            'x': x_coord,
                                            'y': y_coord,
                                            'bbox': span["bbox"]
                                        },
                                        'font_info': {
                                            'size': font_size,
                                            'flags': font_flags,
                                            'font': span.get("font", "")
                                        },
                                        'hierarchy_level': self._determine_hierarchy_level(section_match['id'])
                                    }
                                    
                                    sections.append(section_info)
                                    logger.info(f"Detected section: {section_info['title']} on page {page_num + 1}")
        
        return self._organize_section_hierarchy(sections)
    
    def _estimate_body_font_size(self, fitz_doc: fitz.Document) -> float:
        """Estimate the most common font size (body text) in the document."""
        font_sizes = []
        
        # Sample first few pages to estimate body font size
        pages_to_sample = min(3, len(fitz_doc))
        
        for page_num in range(pages_to_sample):
            page = fitz_doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        if len(span["text"].strip()) > 20:  # Only consider longer text spans
                            font_sizes.append(span["size"])
        
        if font_sizes:
            # Return the most common font size
            font_sizes.sort()
            mid = len(font_sizes) // 2
            return font_sizes[mid]  # Median font size
        
        return 11.0  # Default fallback
    
    def _is_potential_header(self, text: str, font_size: float, font_flags: int, body_font_size: float) -> bool:
        """Check if text could be a section header based on style."""
        if not text or len(text.strip()) < 3:
            return False
            
        # Check font size (should be larger than body text)
        if font_size < body_font_size * self.header_font_threshold:
            return False
            
        # Check for bold formatting (flag 16 indicates bold)
        is_bold = bool(font_flags & 16)
        
        # Headers are typically short (less than 100 characters)
        if len(text) > 100:
            return False
            
        # Must start with capital letter or number
        if not (text[0].isupper() or text[0].isdigit()):
            return False
            
        return True
    
    def _match_section_patterns(self, text: str) -> Optional[Dict[str, str]]:
        """Match text against section patterns and extract section info."""
        for pattern in self.section_patterns:
            match = re.match(pattern, text.strip(), re.IGNORECASE)
            if match:
                # Extract section number if present
                section_id = self._extract_section_id(text)
                title = match.group(1) if match.groups() else text.strip()
                
                return {
                    'id': section_id,
                    'title': title.strip(),
                    'pattern_matched': pattern
                }
        
        return None
    
    def _extract_section_id(self, text: str) -> str:
        """Extract section ID (e.g., '1', '2.1', 'A') from section text."""
        # Look for numbered sections
        match = re.match(r'^(\d+(?:\.\d+)*)', text.strip())
        if match:
            return match.group(1)
        
        # Look for lettered sections
        match = re.match(r'^([A-Z])', text.strip())
        if match:
            return match.group(1)
        
        # Generate ID based on position
        return f"sec_{hash(text) % 1000}"
    
    def _determine_hierarchy_level(self, section_id: str) -> int:
        """Determine hierarchical level of section (1, 2, 3, etc.)."""
        if re.match(r'^\d+$', section_id):  # "1", "2", etc.
            return 1
        elif re.match(r'^\d+\.\d+$', section_id):  # "1.1", "2.3", etc.
            return 2
        elif re.match(r'^\d+\.\d+\.\d+$', section_id):  # "1.1.1", etc.
            return 3
        else:
            return 1  # Default to top level
    
    def _organize_section_hierarchy(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Organize sections into hierarchical structure and add parent-child relationships."""
        # Sort sections by page number and y-coordinate
        sections.sort(key=lambda x: (x['page_num'], x['coordinates']['y']))
        
        # Add parent-child relationships
        for i, section in enumerate(sections):
            section['parent_section'] = None
            section['child_sections'] = []
            
            # Find parent section (previous section with lower hierarchy level)
            current_level = section['hierarchy_level']
            for j in range(i - 1, -1, -1):
                if sections[j]['hierarchy_level'] < current_level:
                    section['parent_section'] = sections[j]['section_id']
                    sections[j]['child_sections'].append(section['section_id'])
                    break
        
        return sections
    
    def _extract_page_info(self, fitz_doc: fitz.Document, pdfplumber_doc) -> List[Dict[str, Any]]:
        """Extract detailed information for each page."""
        pages_info = []
        
        for page_num in range(len(fitz_doc)):
            fitz_page = fitz_doc[page_num]
            pdfplumber_page = pdfplumber_doc.pages[page_num] if page_num < len(pdfplumber_doc.pages) else None
            
            page_info = {
                'page_num': page_num + 1,
                'bbox': fitz_page.rect,
                'rotation': fitz_page.rotation,
                'text_blocks': [],
                'images': [],
                'tables': []
            }
            
            # Extract text blocks with positioning
            blocks = fitz_page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    block_text = ""
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                        block_text += line_text + "\n"
                    
                    if block_text.strip():
                        page_info['text_blocks'].append({
                            'text': block_text.strip(),
                            'bbox': block["bbox"],
                            'block_type': 'text'
                        })
                else:
                    # Image block
                    page_info['images'].append({
                        'bbox': block["bbox"],
                        'block_type': 'image'
                    })
            
            # Extract tables using pdfplumber if available
            if pdfplumber_page:
                tables = pdfplumber_page.find_tables()
                for table in tables:
                    page_info['tables'].append({
                        'bbox': table.bbox,
                        'cells': table.cells if hasattr(table, 'cells') else [],
                        'block_type': 'table'
                    })
            
            pages_info.append(page_info)
        
        return pages_info
    
    def get_text_between_sections(self, doc_info: Dict[str, Any], 
                                 start_section_id: str, 
                                 end_section_id: Optional[str] = None) -> str:
        """
        Extract text between two sections.
        
        Args:
            doc_info: Document information from process_document()
            start_section_id: ID of starting section
            end_section_id: ID of ending section (if None, extract to end of document)
            
        Returns:
            Text content between sections
        """
        sections = doc_info['sections']
        pages = doc_info['pages']
        
        # Find section coordinates
        start_section = None
        end_section = None
        
        for section in sections:
            if section['section_id'] == start_section_id:
                start_section = section
            if end_section_id and section['section_id'] == end_section_id:
                end_section = section
        
        if not start_section:
            return ""
        
        # Extract text from start section to end section (or end of document)
        text_parts = []
        start_page = start_section['page_num'] - 1
        start_y = start_section['coordinates']['y']
        
        end_page = len(pages) - 1
        end_y = float('inf')
        
        if end_section:
            end_page = end_section['page_num'] - 1
            end_y = end_section['coordinates']['y']
        
        for page_num in range(start_page, end_page + 1):
            page_info = pages[page_num]
            
            for block in page_info['text_blocks']:
                block_y = block['bbox'][1]  # y-coordinate of block
                
                # Check if block is within the section range
                if page_num == start_page and block_y < start_y:
                    continue
                if page_num == end_page and block_y >= end_y:
                    continue
                
                text_parts.append(block['text'])
        
        return '\n'.join(text_parts)