"""
Structured Response Formatter

This module provides functionality to format responses with clear sections for answers,
explanations, and well-formatted citations from source documents.
"""

from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FormattedSource:
    """Formatted source information."""
    citation: str
    quote: str
    relevance_score: float
    document_title: str
    page_number: int
    section_title: Optional[str] = None


class ResponseFormatter:
    """
    Formats responses with consistent structure including answer, explanation,
    sources, and confidence levels.
    """
    
    def __init__(self, use_markdown: bool = True, max_sources: int = 5):
        """
        Initialize response formatter.
        
        Args:
            use_markdown: Whether to use markdown formatting
            max_sources: Maximum number of sources to include
        """
        self.use_markdown = use_markdown
        self.max_sources = max_sources
    
    def format_response(self, answer: str, explanation: str, 
                       sources: List[Dict[str, Any]], 
                       confidence: Dict[str, Any]) -> str:
        """
        Format a complete response with all sections.
        
        Args:
            answer: Main answer text
            explanation: Detailed explanation
            sources: List of source information
            confidence: Confidence assessment
            
        Returns:
            Formatted response string
        """
        formatted_parts = []
        
        # Answer section
        formatted_parts.append(self._format_answer_section(answer))
        
        # Explanation section
        if explanation.strip():
            formatted_parts.append(self._format_explanation_section(explanation))
        
        # Sources section
        if sources:
            formatted_parts.append(self._format_sources_section(sources))
        
        # Confidence section
        formatted_parts.append(self._format_confidence_section(confidence))
        
        return "\n\n".join(formatted_parts)
    
    def _format_answer_section(self, answer: str) -> str:
        """Format the main answer section."""
        if self.use_markdown:
            return f"### Ответ\n{answer.strip()}"
        else:
            return f"Ответ:\n{answer.strip()}"
    
    def _format_explanation_section(self, explanation: str) -> str:
        """Format the explanation section."""
        if self.use_markdown:
            return f"### Объяснение\n{explanation.strip()}"
        else:
            return f"Объяснение:\n{explanation.strip()}"
    
    def _format_sources_section(self, sources: List[Dict[str, Any]]) -> str:
        """Format the sources section with citations and quotes."""
        if not sources:
            return ""
        
        # Limit number of sources
        limited_sources = sources[:self.max_sources]
        
        if self.use_markdown:
            header = "### Источники"
        else:
            header = "Источники:"
        
        formatted_sources = []
        for i, source in enumerate(limited_sources, 1):
            formatted_source = self._format_single_source(source, i)
            if formatted_source:
                formatted_sources.append(formatted_source)
        
        if formatted_sources:
            return f"{header}\n" + "\n".join(formatted_sources)
        else:
            return f"{header}\nИсточники недоступны."
    
    def _format_single_source(self, source: Dict[str, Any], index: int) -> str:
        """Format a single source with citation and quote."""
        # Extract source information
        citation = source.get('citation', '')
        quote = source.get('quote', '')
        document_title = source.get('document_title', '')
        page_number = source.get('page_number', '')
        section_title = source.get('section_title', '')
        
        # Build citation string
        if citation:
            citation_text = citation
        else:
            # Construct citation from components
            citation_parts = []
            if document_title:
                citation_parts.append(document_title)
            if page_number:
                citation_parts.append(f"стр. {page_number}")
            if section_title:
                citation_parts.append(section_title)
            citation_text = ", ".join(citation_parts)
        
        if not citation_text:
            return ""
        
        # Format source entry
        if self.use_markdown:
            formatted = f"{index}. **{citation_text}**"
        else:
            formatted = f"{index}. {citation_text}"
        
        # Add quote if available
        if quote and quote.strip():
            cleaned_quote = self._clean_quote(quote)
            if self.use_markdown:
                formatted += f"\n   > \"{cleaned_quote}\""
            else:
                formatted += f"\n   \"{cleaned_quote}\""
        
        return formatted
    
    def _clean_quote(self, quote: str) -> str:
        """Clean and format quote text."""
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', quote.strip())
        
        # Limit quote length
        max_quote_length = 300
        if len(cleaned) > max_quote_length:
            # Try to break at sentence boundary
            truncated = cleaned[:max_quote_length]
            last_sentence = max(
                truncated.rfind('.'),
                truncated.rfind('!'),
                truncated.rfind('?')
            )
            
            if last_sentence > max_quote_length * 0.7:
                cleaned = truncated[:last_sentence + 1]
            else:
                # Break at word boundary
                last_space = truncated.rfind(' ')
                if last_space > 0:
                    cleaned = truncated[:last_space] + "..."
                else:
                    cleaned = truncated + "..."
        
        return cleaned
    
    def _format_confidence_section(self, confidence: Dict[str, Any]) -> str:
        """Format the confidence section."""
        level = confidence.get('level', 'Unknown')
        explanation = confidence.get('explanation', '')
        score = confidence.get('score', 0.0)
        
        if self.use_markdown:
            header = f"### Уровень уверенности: {level}"
        else:
            header = f"Уровень уверенности: {level}"
        
        content_parts = [header]
        
        if explanation:
            content_parts.append(explanation)
        
        # Add score if available (for debugging/analysis)
        if score > 0:
            score_text = f"Оценка: {score:.2f}/1.00"
            if self.use_markdown:
                content_parts.append(f"*{score_text}*")
            else:
                content_parts.append(f"({score_text})")
        
        return "\n".join(content_parts)
    
    def format_citation_with_quote(self, citation_data: Dict[str, Any]) -> str:
        """
        Format a single citation with quote for use in other contexts.
        
        Args:
            citation_data: Citation information
            
        Returns:
            Formatted citation string
        """
        document_title = citation_data.get('document_title', 'Unknown Document')
        page_number = citation_data.get('page_number', '')
        section_title = citation_data.get('section_title', '')
        quote = citation_data.get('quote', citation_data.get('text_snippet', ''))
        
        # Build citation
        citation_parts = [document_title]
        if page_number:
            citation_parts.append(f"стр. {page_number}")
        if section_title:
            citation_parts.append(section_title)
        
        citation_text = ", ".join(citation_parts)
        
        if quote:
            cleaned_quote = self._clean_quote(quote)
            if self.use_markdown:
                return f"**{citation_text}** - \"{cleaned_quote}\""
            else:
                return f"{citation_text} - \"{cleaned_quote}\""
        else:
            if self.use_markdown:
                return f"**{citation_text}**"
            else:
                return citation_text
    
    def calculate_confidence(self, retrieval_scores: List[float], 
                           answer_consistency: float = 0.8) -> Dict[str, Any]:
        """
        Calculate confidence level based on retrieval scores and consistency.
        
        Args:
            retrieval_scores: List of retrieval relevance scores
            answer_consistency: Consistency score for the answer
            
        Returns:
            Confidence assessment dictionary
        """
        if not retrieval_scores:
            return {
                "level": "Very Low",
                "score": 0.0,
                "explanation": "No sources available for verification."
            }
        
        # Calculate average retrieval score
        avg_score = sum(retrieval_scores) / len(retrieval_scores)
        
        # Normalize scores to 0-1 range if needed
        if avg_score > 1.0:
            avg_score = avg_score / max(retrieval_scores) if max(retrieval_scores) > 0 else 0.0
        
        # Combine with answer consistency
        final_score = (avg_score * 0.7) + (answer_consistency * 0.3)
        
        # Determine confidence level and explanation
        if final_score > 0.8:
            level = "High"
            explanation = "Multiple high-quality sources provide consistent and reliable information."
        elif final_score > 0.6:
            level = "Medium"
            explanation = "Sources provide good information with reasonable confidence in the answer."
        elif final_score > 0.4:
            level = "Low"
            explanation = "Limited source quality or consistency, answer should be verified."
        else:
            level = "Very Low"
            explanation = "Insufficient or low-quality sources, answer reliability is questionable."
        
        return {
            "level": level,
            "score": final_score,
            "explanation": explanation
        }
    
    def format_for_display(self, response_dict: Dict[str, Any]) -> str:
        """
        Format a response dictionary for display.
        
        Args:
            response_dict: Dictionary with answer, explanation, sources, confidence
            
        Returns:
            Formatted response string
        """
        answer = response_dict.get('answer', '')
        explanation = response_dict.get('explanation', '')
        sources = response_dict.get('sources', [])
        confidence = response_dict.get('confidence', {})
        
        return self.format_response(answer, explanation, sources, confidence)
    
    def extract_key_points(self, explanation: str) -> List[str]:
        """
        Extract key points from explanation for summary.
        
        Args:
            explanation: Explanation text
            
        Returns:
            List of key points
        """
        if not explanation:
            return []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', explanation)
        
        # Filter and clean sentences
        key_points = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Ignore very short sentences
                # Remove leading conjunctions
                sentence = re.sub(r'^(However|Moreover|Furthermore|Additionally|Therefore|Thus),?\s*', '', sentence)
                if sentence:
                    key_points.append(sentence)
        
        # Return first few key points
        return key_points[:3]
    
    def create_summary_response(self, full_response: str) -> str:
        """
        Create a shortened summary version of a response.
        
        Args:
            full_response: Full formatted response
            
        Returns:
            Summary version
        """
        # Extract just the answer section
        answer_match = re.search(r'### Ответ\n(.*?)(?=\n### |$)', full_response, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            # Fallback: look for non-markdown format
            answer_match = re.search(r'Ответ:\n(.*?)(?=\nОбъяснение:|$)', full_response, re.DOTALL)
            answer = answer_match.group(1).strip() if answer_match else "Ответ недоступен."
        
        # Extract confidence level
        confidence_match = re.search(r'Уровень уверенности: (\w+)', full_response)
        confidence_level = confidence_match.group(1) if confidence_match else "Unknown"
        
        # Create summary
        if self.use_markdown:
            return f"**Краткий ответ:** {answer}\n\n*Уровень уверенности: {confidence_level}*"
        else:
            return f"Краткий ответ: {answer}\n\nУровень уверенности: {confidence_level}"