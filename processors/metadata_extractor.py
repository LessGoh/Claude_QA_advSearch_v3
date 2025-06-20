"""
Metadata Extractor for Academic Financial Papers

This module provides functionality to extract metadata from academic papers including
title, authors, keywords, publication information, and other relevant metadata.
"""

import re
import fitz  # pymupdf
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Extracts comprehensive metadata from academic financial papers.
    """
    
    def __init__(self):
        # Patterns for metadata extraction
        self.title_patterns = [
            r'^([A-Z][^\n]{10,200})$',  # Title case, reasonable length
            r'(?:TITLE|Title):\s*([^\n]+)',  # Explicit title label
            r'^([A-Z][A-Z\s]{5,100})$',  # All caps title
        ]
        
        self.author_patterns = [
            r'(?:Author|AUTHORS?|By)s?\s*:\s*([^\n]+)',  # Explicit author label
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?)+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?)+)*)\s*$',  # Name patterns
            r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+\s+[A-Z][a-z]+)*)',  # Simple name pattern
        ]
        
        self.keyword_patterns = [
            r'(?:Keywords?|KEY\s*WORDS?):\s*([^\n]+)',
            r'(?:JEL\s*Classification|JEL\s*Codes?):\s*([^\n]+)',  # Financial papers often use JEL codes
        ]
        
        self.abstract_patterns = [
            r'(?:ABSTRACT|Abstract)\s*[:.]?\s*\n(.*?)(?:\n\s*(?:Keywords?|JEL|1\.|Introduction|INTRODUCTION))',
            r'(?:ABSTRACT|Abstract)\s*[:.]?\s*(.*?)(?:Keywords?|JEL|1\.|Introduction)',
        ]
        
        self.publication_patterns = [
            r'(?:Journal|Published\s+in|Forthcoming\s+in):\s*([^\n]+)',
            r'([A-Z][a-z]+\s+(?:Journal|Review|of).*?)\s*,?\s*(?:Vol|Volume)',
            r'(?:DOI|doi):\s*(10\.\d+/[^\s]+)',
        ]
        
        self.date_patterns = [
            r'(?:Date|Published|Revised):\s*([A-Z][a-z]+\s+\d{4})',
            r'(?:Date|Published|Revised):\s*(\d{1,2}/\d{1,2}/\d{4})',
            r'(?:Date|Published|Revised):\s*(\d{4}-\d{2}-\d{2})',
            r'(\d{4})',  # Simple year
        ]
        
        # Financial domain keywords
        self.financial_keywords = [
            'asset pricing', 'capm', 'volatility', 'portfolio', 'options', 'derivatives',
            'risk management', 'market efficiency', 'behavioral finance', 'corporate finance',
            'capital structure', 'dividend policy', 'merger', 'acquisition', 'ipo',
            'credit risk', 'liquidity', 'hedge fund', 'mutual fund', 'pension fund',
            'interest rate', 'bond', 'equity', 'stock', 'return', 'alpha', 'beta',
            'sharpe ratio', 'var', 'stress test', 'basel', 'regulation', 'compliance'
        ]
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from PDF document.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing extracted metadata
        """
        logger.info(f"Extracting metadata from: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            
            # Extract metadata
            metadata = {
                'file_path': pdf_path,
                'extraction_date': datetime.now().isoformat(),
                'basic_metadata': self._extract_basic_metadata(doc),
                'title': self._extract_title(doc),
                'authors': self._extract_authors(doc),
                'abstract': self._extract_abstract(doc),
                'keywords': self._extract_keywords(doc),
                'publication_info': self._extract_publication_info(doc),
                'date': self._extract_date(doc),
                'financial_indicators': self._detect_financial_content(doc),
                'document_stats': self._calculate_document_stats(doc)
            }
            
            doc.close()
            logger.info(f"Metadata extraction completed for: {pdf_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {str(e)}")
            raise
    
    def _extract_basic_metadata(self, doc: fitz.Document) -> Dict[str, str]:
        """Extract basic metadata from PDF properties."""
        metadata = doc.metadata
        return {
            'pdf_title': metadata.get('title', ''),
            'pdf_author': metadata.get('author', ''),
            'pdf_subject': metadata.get('subject', ''),
            'pdf_creator': metadata.get('creator', ''),
            'pdf_producer': metadata.get('producer', ''),
            'creation_date': metadata.get('creationDate', ''),
            'modification_date': metadata.get('modDate', ''),
            'page_count': len(doc)
        }
    
    def _extract_title(self, doc: fitz.Document) -> Optional[str]:
        """Extract paper title from first few pages."""
        # Check first 3 pages for title
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            
            # Look for largest font size text (likely title)
            title_candidate = self._find_title_by_font_size(text_dict)
            if title_candidate:
                return title_candidate
            
            # Try pattern matching
            page_text = page.get_text()
            for pattern in self.title_patterns:
                match = re.search(pattern, page_text, re.MULTILINE | re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    if self._is_valid_title(candidate):
                        return candidate
        
        # Fallback to PDF metadata title
        pdf_title = doc.metadata.get('title', '')
        if pdf_title and self._is_valid_title(pdf_title):
            return pdf_title
        
        return None
    
    def _find_title_by_font_size(self, text_dict: Dict) -> Optional[str]:
        """Find title based on font size analysis."""
        font_sizes = []
        text_by_font_size = {}
        
        # Collect all font sizes and associated text
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
            
            for line in block["lines"]:
                for span in line["spans"]:
                    font_size = span["size"]
                    text = span["text"].strip()
                    
                    if text and len(text) > 10:  # Ignore very short text
                        font_sizes.append(font_size)
                        if font_size not in text_by_font_size:
                            text_by_font_size[font_size] = []
                        text_by_font_size[font_size].append(text)
        
        if not font_sizes:
            return None
        
        # Find the largest font size
        max_font_size = max(font_sizes)
        title_candidates = text_by_font_size.get(max_font_size, [])
        
        # Return the longest text with the largest font size
        if title_candidates:
            title = max(title_candidates, key=len)
            if self._is_valid_title(title):
                return title
        
        return None
    
    def _is_valid_title(self, text: str) -> bool:
        """Check if text is a valid title."""
        if not text or len(text) < 10 or len(text) > 300:
            return False
        
        # Should not be all caps unless it's a reasonable length
        if text.isupper() and len(text) > 100:
            return False
        
        # Should not contain too many special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and c not in ' :-.,()[]') / len(text)
        if special_char_ratio > 0.3:
            return False
        
        return True
    
    def _extract_authors(self, doc: fitz.Document) -> List[str]:
        """Extract author names from the document."""
        authors = []
        
        # Check first few pages for author information
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            page_text = page.get_text()
            
            for pattern in self.author_patterns:
                matches = re.findall(pattern, page_text, re.MULTILINE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    
                    # Split multiple authors
                    author_names = self._parse_author_names(match)
                    authors.extend(author_names)
        
        # Remove duplicates and clean up
        unique_authors = []
        for author in authors:
            cleaned = self._clean_author_name(author)
            if cleaned and cleaned not in unique_authors and self._is_valid_author_name(cleaned):
                unique_authors.append(cleaned)
        
        return unique_authors[:10]  # Limit to reasonable number
    
    def _parse_author_names(self, author_text: str) -> List[str]:
        """Parse multiple author names from text."""
        # Split by common separators
        separators = [',', ' and ', '&', '\\n']
        authors = [author_text]
        
        for sep in separators:
            new_authors = []
            for author in authors:
                new_authors.extend(author.split(sep))
            authors = new_authors
        
        return [author.strip() for author in authors if author.strip()]
    
    def _clean_author_name(self, name: str) -> str:
        """Clean and normalize author name."""
        # Remove common suffixes and prefixes
        name = re.sub(r'\b(?:Dr|Prof|Professor|PhD|Ph\.D\.)\b\.?', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\d+', '', name)  # Remove numbers
        name = re.sub(r'\s+', ' ', name).strip()  # Normalize whitespace
        return name
    
    def _is_valid_author_name(self, name: str) -> bool:
        """Check if text is a valid author name."""
        if not name or len(name) < 5 or len(name) > 50:
            return False
        
        # Should contain at least first and last name
        words = name.split()
        if len(words) < 2:
            return False
        
        # Should not contain too many special characters
        if sum(1 for c in name if not c.isalpha() and c not in ' .-') > 3:
            return False
        
        return True
    
    def _extract_abstract(self, doc: fitz.Document) -> Optional[str]:
        """Extract abstract from the document."""
        # Check first few pages for abstract
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            page_text = page.get_text()
            
            for pattern in self.abstract_patterns:
                match = re.search(pattern, page_text, re.MULTILINE | re.IGNORECASE | re.DOTALL)
                if match:
                    abstract = match.group(1).strip()
                    if self._is_valid_abstract(abstract):
                        return abstract
        
        return None
    
    def _is_valid_abstract(self, text: str) -> bool:
        """Check if text is a valid abstract."""
        if not text or len(text) < 50 or len(text) > 2000:
            return False
        
        # Should contain multiple sentences
        sentences = text.split('.')
        if len(sentences) < 3:
            return False
        
        return True
    
    def _extract_keywords(self, doc: fitz.Document) -> List[str]:
        """Extract keywords from the document."""
        keywords = []
        
        # Check first few pages for keywords
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            page_text = page.get_text()
            
            for pattern in self.keyword_patterns:
                match = re.search(pattern, page_text, re.MULTILINE | re.IGNORECASE)
                if match:
                    keyword_text = match.group(1).strip()
                    # Split keywords by common separators
                    keyword_list = re.split(r'[,;]', keyword_text)
                    keywords.extend([kw.strip() for kw in keyword_list if kw.strip()])
        
        # Also detect financial keywords from content
        full_text = ""
        for page_num in range(min(5, len(doc))):
            full_text += doc[page_num].get_text().lower()
        
        detected_keywords = []
        for keyword in self.financial_keywords:
            if keyword.lower() in full_text:
                detected_keywords.append(keyword)
        
        # Combine and deduplicate
        all_keywords = keywords + detected_keywords
        unique_keywords = []
        for kw in all_keywords:
            if kw not in unique_keywords and len(kw) > 2:
                unique_keywords.append(kw)
        
        return unique_keywords[:20]  # Limit to reasonable number
    
    def _extract_publication_info(self, doc: fitz.Document) -> Dict[str, Optional[str]]:
        """Extract publication information."""
        pub_info = {
            'journal': None,
            'volume': None,
            'issue': None,
            'pages': None,
            'doi': None,
            'publisher': None
        }
        
        # Check first few pages
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            page_text = page.get_text()
            
            for pattern in self.publication_patterns:
                match = re.search(pattern, page_text, re.MULTILINE | re.IGNORECASE)
                if match:
                    if 'doi' in pattern.lower():
                        pub_info['doi'] = match.group(1)
                    else:
                        pub_info['journal'] = match.group(1)
        
        return pub_info
    
    def _extract_date(self, doc: fitz.Document) -> Optional[str]:
        """Extract publication or creation date."""
        # Check document metadata first
        creation_date = doc.metadata.get('creationDate', '')
        if creation_date:
            return creation_date
        
        # Check document content
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            page_text = page.get_text()
            
            for pattern in self.date_patterns:
                match = re.search(pattern, page_text)
                if match:
                    return match.group(1)
        
        return None
    
    def _detect_financial_content(self, doc: fitz.Document) -> Dict[str, Any]:
        """Detect financial content indicators."""
        full_text = ""
        for page_num in range(min(10, len(doc))):  # Check first 10 pages
            full_text += doc[page_num].get_text().lower()
        
        indicators = {
            'has_mathematical_formulas': bool(re.search(r'\$.*?\$|\\[a-zA-Z]+\{.*?\}', full_text)),
            'has_financial_terms': False,
            'financial_term_count': 0,
            'has_tables': bool(re.search(r'table\s+\d+|figure\s+\d+', full_text, re.IGNORECASE)),
            'has_references': bool(re.search(r'references|bibliography', full_text, re.IGNORECASE)),
            'estimated_academic_level': 'research'
        }
        
        # Count financial terms
        financial_term_count = 0
        for term in self.financial_keywords:
            if term.lower() in full_text:
                financial_term_count += full_text.count(term.lower())
        
        indicators['financial_term_count'] = financial_term_count
        indicators['has_financial_terms'] = financial_term_count > 5
        
        return indicators
    
    def _calculate_document_stats(self, doc: fitz.Document) -> Dict[str, Any]:
        """Calculate document statistics."""
        total_chars = 0
        total_words = 0
        
        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text()
            total_chars += len(page_text)
            total_words += len(page_text.split())
        
        return {
            'page_count': len(doc),
            'total_characters': total_chars,
            'total_words': total_words,
            'avg_words_per_page': total_words / len(doc) if len(doc) > 0 else 0,
            'estimated_reading_time_minutes': total_words / 200  # Assuming 200 words per minute
        }