"""
Smart Chunking Algorithm for Academic Financial Documents

This module provides intelligent chunking that preserves mathematical formulas,
tables, and semantic coherence while respecting document structure.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
import logging
import hashlib
from dataclasses import dataclass

# Import spacy conditionally
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata for document chunks."""
    chunk_id: str
    content_type: str  # 'text', 'formula', 'table', 'figure'
    page_num: int
    section_id: Optional[str]
    start_y: float
    end_y: float
    placeholders: List[str]
    formula_count: int = 0
    table_count: int = 0


class SmartChunker:
    """
    Advanced chunking algorithm that preserves document structure and content integrity.
    """
    
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.semantic_threshold = 0.7
        
        # Load spaCy model with fallback to simple processing
        self.nlp = None
        self.use_spacy = False
        
        if SPACY_AVAILABLE and spacy:
            try:
                self.nlp = spacy.load("en_core_web_lg")
                self.use_spacy = True
                logger.info("Using spaCy model: en_core_web_lg")
            except OSError:
                logger.warning("en_core_web_lg not found, falling back to en_core_web_sm")
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    self.use_spacy = True
                    logger.info("Using spaCy model: en_core_web_sm")
                except OSError:
                    logger.warning("No spaCy model found. Using simple text processing fallback.")
                    self.use_spacy = False
        else:
            logger.info("spaCy not available. Using simple text processing fallback.")
        
        # Formula detection patterns
        self.formula_patterns = [
            r'\$\$.+?\$\$',  # Display math: $$formula$$
            r'\\begin\{equation\}.+?\\end\{equation\}',  # LaTeX equation environments
            r'\\begin\{align\}.+?\\end\{align\}',  # LaTeX align environments
            r'\\\[.+?\\\]',  # Alternative display math: \[ formula \]
            r'\\\(.+?\\\)',  # Alternative inline math: \( formula \)
            r'\$[^$\n]{2,50}\$',  # Inline math: $formula$ (avoiding currency)
        ]
        
        # Currency pattern to avoid false formula detection
        self.currency_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|billion|trillion|thousand|k|m|b))?'
        
        # Table detection patterns
        self.table_patterns = [
            r'(?:Table\s+\d+|TABLE\s+\d+)',
            r'(?:Figure\s+\d+|FIGURE\s+\d+)',
        ]
        
        # Initialize placeholder counter
        self.placeholder_counter = 0
    
    def chunk_document(self, document_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk document using smart algorithm that preserves content integrity.
        
        Args:
            document_data: Document data from EnhancedPDFProcessor
            
        Returns:
            List of chunks with metadata
        """
        logger.info("Starting smart chunking process")
        
        chunks = []
        self.placeholder_counter = 0
        
        # Process each section
        sections = document_data.get('sections', [])
        pages = document_data.get('pages', [])
        
        if not sections:
            # Fallback: process entire document as one section
            full_text = self._extract_full_text(pages)
            section_chunks = self._chunk_section(full_text, None, pages)
            chunks.extend(section_chunks)
        else:
            for section in sections:
                section_text = self._extract_section_text(section, document_data)
                if section_text.strip():
                    section_chunks = self._chunk_section(section_text, section, pages)
                    chunks.extend(section_chunks)
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def _extract_full_text(self, pages: List[Dict[str, Any]]) -> str:
        """Extract all text from document pages."""
        text_parts = []
        for page in pages:
            for block in page.get('text_blocks', []):
                text_parts.append(block['text'])
        return '\n'.join(text_parts)
    
    def _extract_section_text(self, section: Dict[str, Any], document_data: Dict[str, Any]) -> str:
        """Extract text for a specific section."""
        # Use the enhanced PDF processor method if available
        if hasattr(self, '_processor'):
            return self._processor.get_text_between_sections(
                document_data, 
                section['section_id']
            )
        
        # Fallback: extract based on coordinates
        pages = document_data.get('pages', [])
        section_text = []
        
        start_page = section['page_num'] - 1
        start_y = section['coordinates']['y']
        
        # Find next section to determine end point
        sections = document_data.get('sections', [])
        next_section = None
        for s in sections:
            if (s['page_num'] > section['page_num'] or 
                (s['page_num'] == section['page_num'] and s['coordinates']['y'] > start_y)):
                if not next_section or s['coordinates']['y'] < next_section['coordinates']['y']:
                    next_section = s
        
        end_page = len(pages) - 1
        end_y = float('inf')
        
        if next_section:
            end_page = next_section['page_num'] - 1
            end_y = next_section['coordinates']['y']
        
        # Extract text from the section range
        for page_num in range(start_page, min(end_page + 1, len(pages))):
            page = pages[page_num]
            
            for block in page.get('text_blocks', []):
                block_y = block['bbox'][1] if isinstance(block['bbox'], (list, tuple)) else 0
                
                # Check if block is within section range
                if page_num == start_page and block_y < start_y:
                    continue
                if page_num == end_page and block_y >= end_y:
                    continue
                
                section_text.append(block['text'])
        
        return '\n'.join(section_text)
    
    def _chunk_section(self, text: str, section: Optional[Dict[str, Any]], 
                      pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a section of text using intelligent algorithms.
        
        Args:
            text: Section text to chunk
            section: Section metadata
            pages: Page information for coordinate lookup
            
        Returns:
            List of chunks for this section
        """
        if not text.strip():
            return []
        
        # Step 1: Preserve special content (formulas and tables)
        preserved_text, preservation_map = self._preserve_special_content(text)
        
        # Step 2: Split into paragraphs
        paragraphs = self._split_into_paragraphs(preserved_text)
        
        # Step 3: Group related paragraphs semantically
        paragraph_groups = self._group_paragraphs_semantically(paragraphs)
        
        # Step 4: Create adaptive chunks
        chunks = self._create_adaptive_chunks(paragraph_groups, preservation_map, section)
        
        return chunks
    
    def _preserve_special_content(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Replace formulas and tables with placeholders and store original content.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (modified_text, preservation_map)
        """
        preservation_map = {}
        modified_text = text
        
        # First, protect currency values from being detected as formulas
        currency_matches = list(re.finditer(self.currency_pattern, text, re.IGNORECASE))
        currency_placeholders = {}
        
        for i, match in enumerate(currency_matches):
            currency_id = f"__CURRENCY_{i}__"
            currency_placeholders[currency_id] = match.group(0)
            modified_text = modified_text.replace(match.group(0), currency_id, 1)
        
        # Preserve formulas
        for pattern in self.formula_patterns:
            matches = list(re.finditer(pattern, modified_text, re.DOTALL))
            for match in matches:
                formula_content = match.group(0)
                
                # Skip if this looks like currency that was temporarily replaced
                if any(currency_id in formula_content for currency_id in currency_placeholders):
                    continue
                
                placeholder_id = f"__FORMULA_{self.placeholder_counter}__"
                self.placeholder_counter += 1
                
                preservation_map[placeholder_id] = {
                    'type': 'formula',
                    'content': formula_content,
                    'normalized': self._normalize_formula(formula_content)
                }
                
                modified_text = modified_text.replace(formula_content, placeholder_id, 1)
        
        # Preserve table references and captions
        for pattern in self.table_patterns:
            matches = list(re.finditer(pattern, modified_text, re.IGNORECASE))
            for match in matches:
                # Look for table caption (usually follows the table reference)
                start_pos = match.end()
                end_pos = min(start_pos + 200, len(modified_text))
                
                # Try to find the end of the caption (usually at next paragraph or section)
                caption_text = modified_text[match.start():end_pos]
                caption_end = re.search(r'\n\s*\n|\n\s*[A-Z]', caption_text[50:])  # Skip first 50 chars
                
                if caption_end:
                    caption_text = caption_text[:50 + caption_end.start()]
                
                placeholder_id = f"__TABLE_{self.placeholder_counter}__"
                self.placeholder_counter += 1
                
                preservation_map[placeholder_id] = {
                    'type': 'table',
                    'content': caption_text,
                    'reference': match.group(0)
                }
                
                modified_text = modified_text.replace(caption_text, placeholder_id, 1)
        
        # Restore currency values
        for currency_id, currency_value in currency_placeholders.items():
            modified_text = modified_text.replace(currency_id, currency_value)
        
        return modified_text, preservation_map
    
    def _normalize_formula(self, formula: str) -> str:
        """Normalize formula for better searching."""
        # Remove delimiters
        normalized = formula
        for delim in ['$$', '$', '\\[', '\\]', '\\(', '\\)', 
                     '\\begin{equation}', '\\end{equation}',
                     '\\begin{align}', '\\end{align}']:
            normalized = normalized.replace(delim, '')
        
        # Normalize spacing
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Replace common LaTeX commands with text
        replacements = {
            '\\alpha': 'alpha',
            '\\beta': 'beta', 
            '\\sigma': 'sigma',
            '\\mu': 'mu',
            '\\sum': 'sum',
            '\\int': 'integral',
            '\\frac': 'fraction',
            '\\sqrt': 'square_root'
        }
        
        for latex_cmd, text_rep in replacements.items():
            normalized = normalized.replace(latex_cmd, text_rep)
        
        return normalized
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into logical paragraphs."""
        # Split on double newlines first
        initial_split = re.split(r'\n\s*\n', text)
        
        paragraphs = []
        for chunk in initial_split:
            chunk = chunk.strip()
            if not chunk:
                continue
            
            # Further split very long paragraphs at sentence boundaries
            if len(chunk) > self.max_chunk_size * 1.5:
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                current_para = ""
                
                for sentence in sentences:
                    if len(current_para + sentence) < self.max_chunk_size:
                        current_para += sentence + " "
                    else:
                        if current_para.strip():
                            paragraphs.append(current_para.strip())
                        current_para = sentence + " "
                
                if current_para.strip():
                    paragraphs.append(current_para.strip())
            else:
                paragraphs.append(chunk)
        
        return paragraphs
    
    def _group_paragraphs_semantically(self, paragraphs: List[str]) -> List[List[str]]:
        """Group related paragraphs based on semantic similarity."""
        if not paragraphs:
            return []
        
        groups = [[paragraphs[0]]]
        
        for i in range(1, len(paragraphs)):
            current_para = paragraphs[i]
            previous_para = paragraphs[i-1]
            
            # Check if paragraphs should be grouped
            if self._should_group_paragraphs(previous_para, current_para):
                # Add to current group
                groups[-1].append(current_para)
            else:
                # Start new group
                groups.append([current_para])
        
        return groups
    
    def _should_group_paragraphs(self, para1: str, para2: str) -> bool:
        """Determine if two paragraphs should be grouped together."""
        # Skip grouping if paragraphs are too long
        if len(para1) + len(para2) > self.max_chunk_size * 0.8:
            return False
        
        # Don't group if either paragraph has special content placeholders
        placeholders = [p for p in re.findall(r'__(?:FORMULA|TABLE)_\d+__', para1 + para2)]
        if len(placeholders) > 1:  # Multiple special elements
            return False
        
        # Use spaCy if available, otherwise use simple heuristics
        if self.use_spacy and self.nlp:
            try:
                # Use spaCy for semantic similarity
                doc1 = self.nlp(para1[:500])  # Limit text for efficiency
                doc2 = self.nlp(para2[:500])
                
                # Calculate semantic similarity
                similarity = doc1.similarity(doc2)
                
                # Group if similarity is above threshold
                return similarity >= self.semantic_threshold
                
            except Exception as e:
                logger.warning(f"Error calculating semantic similarity: {e}")
                # Fallback to simple heuristics
                return self._simple_grouping_heuristic(para1, para2)
        else:
            # Use simple heuristics when spaCy is not available
            return self._simple_grouping_heuristic(para1, para2)
    
    def _simple_grouping_heuristic(self, para1: str, para2: str) -> bool:
        """Simple fallback grouping logic."""
        # Group if they share common important words
        words1 = set(re.findall(r'\b[a-zA-Z]{4,}\b', para1.lower()))
        words2 = set(re.findall(r'\b[a-zA-Z]{4,}\b', para2.lower()))
        
        if not words1 or not words2:
            return False
        
        # Calculate word overlap
        common_words = words1.intersection(words2)
        overlap_ratio = len(common_words) / min(len(words1), len(words2))
        
        return overlap_ratio >= 0.3
    
    def _create_adaptive_chunks(self, paragraph_groups: List[List[str]], 
                              preservation_map: Dict[str, Dict], 
                              section: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create final chunks with adaptive sizing."""
        chunks = []
        
        for group_idx, paragraph_group in enumerate(paragraph_groups):
            group_text = '\n\n'.join(paragraph_group)
            
            # Check if group fits in one chunk
            if len(group_text) <= self.max_chunk_size:
                chunk = self._create_chunk(group_text, preservation_map, section, group_idx)
                chunks.append(chunk)
            else:
                # Split large groups while preserving paragraph boundaries
                sub_chunks = self._split_large_group(paragraph_group, preservation_map, section, group_idx)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _create_chunk(self, text: str, preservation_map: Dict[str, Dict], 
                     section: Optional[Dict[str, Any]], group_idx: int) -> Dict[str, Any]:
        """Create a single chunk with metadata."""
        # Restore special content
        restored_text = self._restore_special_content(text, preservation_map)
        
        # Find placeholders in this chunk
        placeholders = re.findall(r'__(?:FORMULA|TABLE)_\d+__', text)
        
        # Count special content types
        formula_count = len([p for p in placeholders if 'FORMULA' in p])
        table_count = len([p for p in placeholders if 'TABLE' in p])
        
        # Generate unique chunk ID
        chunk_id = self._generate_chunk_id(restored_text)
        
        # Determine content type
        content_type = 'text'
        if formula_count > 0:
            content_type = 'formula'
        elif table_count > 0:
            content_type = 'table'
        
        chunk_data = {
            'chunk_id': chunk_id,
            'content': restored_text,
            'content_type': content_type,
            'page_num': section['page_num'] if section else 1,
            'section_id': section['section_id'] if section else None,
            'section_title': section['title'] if section else None,
            'placeholders': placeholders,
            'formula_count': formula_count,
            'table_count': table_count,
            'character_count': len(restored_text),
            'word_count': len(restored_text.split()),
            'group_index': group_idx
        }
        
        return chunk_data
    
    def _split_large_group(self, paragraphs: List[str], preservation_map: Dict[str, Dict],
                          section: Optional[Dict[str, Any]], group_idx: int) -> List[Dict[str, Any]]:
        """Split a large paragraph group into multiple chunks."""
        chunks = []
        current_chunk_text = ""
        current_paragraphs = []
        
        for para in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk_text + para) > self.max_chunk_size and current_chunk_text:
                # Create chunk with current content
                chunk = self._create_chunk(current_chunk_text, preservation_map, section, group_idx)
                chunks.append(chunk)
                
                # Start new chunk with overlap if possible
                if current_paragraphs and len(current_paragraphs[-1]) < self.overlap:
                    current_chunk_text = current_paragraphs[-1] + "\n\n" + para
                    current_paragraphs = [current_paragraphs[-1], para]
                else:
                    current_chunk_text = para
                    current_paragraphs = [para]
            else:
                # Add paragraph to current chunk
                if current_chunk_text:
                    current_chunk_text += "\n\n" + para
                else:
                    current_chunk_text = para
                current_paragraphs.append(para)
        
        # Create final chunk if there's remaining content
        if current_chunk_text.strip():
            chunk = self._create_chunk(current_chunk_text, preservation_map, section, group_idx)
            chunks.append(chunk)
        
        return chunks
    
    def _restore_special_content(self, text: str, preservation_map: Dict[str, Dict]) -> str:
        """Restore formulas and tables from placeholders."""
        restored_text = text
        
        for placeholder_id, content_data in preservation_map.items():
            if placeholder_id in restored_text:
                original_content = content_data['content']
                restored_text = restored_text.replace(placeholder_id, original_content)
        
        return restored_text
    
    def _generate_chunk_id(self, text: str) -> str:
        """Generate unique ID for chunk based on content."""
        # Create hash of content for unique ID
        content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
        return f"chunk_{content_hash}"