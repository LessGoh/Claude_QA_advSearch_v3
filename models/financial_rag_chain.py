"""
Financial RAG Chain with Enhanced Prompting

This module implements an enhanced RAG chain specifically designed for academic
financial documents with sophisticated prompting and confidence scoring.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import re
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Response from the Financial RAG Chain."""
    answer: str
    explanation: str
    sources: List[Dict[str, Any]]
    confidence: Dict[str, Any]
    raw_context: str


class FinancialRAGChain:
    """
    Enhanced RAG chain for financial document question answering with
    specialized prompting and confidence assessment.
    """
    
    def __init__(self, llm_client=None, temperature: float = 0.0, max_tokens: int = 1000):
        """
        Initialize the Financial RAG Chain.
        
        Args:
            llm_client: Language model client (OpenAI, etc.)
            temperature: Temperature for text generation
            max_tokens: Maximum tokens for response
        """
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # System prompt defining the AI's role as financial expert
        self.system_prompt = """
You are an expert financial analyst and research assistant with deep knowledge of academic finance literature. Your task is to provide accurate, well-sourced answers to questions about academic financial papers.

When answering:
1. Focus on precise financial terminology and concepts
2. Maintain mathematical accuracy in formulas and equations
3. Cite specific sections, page numbers, and tables from the source documents
4. Present numerical data accurately with proper context
5. Indicate your confidence level in the answer
6. Structure complex answers with clear organization

Always provide citations in the format: [Document Title, Page X, Section Y]
Respond in Russian when the question is asked in Russian, but maintain professional financial terminology.
"""
        
        # Few-shot example for response formatting
        self.few_shot_example = """
Here's an example of a perfectly formatted answer:

Question: What is the expected return according to CAPM?

Answer: The capital asset pricing model (CAPM) estimates a 12.5% expected return for the stock.

Explanation: The CAPM formula calculates expected return as Rf + β(Rm - Rf). With a risk-free rate of 3%, market risk premium of 7%, and beta of 1.35, the calculation is: 3% + 1.35(7%) = 12.45%, rounded to 12.5%. This indicates the stock has above-market risk requiring higher expected returns.

Sources:
1. [Modern Portfolio Theory, Page 42, Section 3.2] - "The CAPM formula expresses expected return as Rf + β(Rm - Rf), where β represents systematic risk."
2. [Market Risk Analysis, Page 87, Table 4.3] - Contains historical beta values showing similar companies in the 1.2-1.5 range.

Confidence: High - The calculation follows standard CAPM methodology with reliable inputs from authoritative sources.
"""
        
        # Financial term patterns for validation
        self.financial_terms = [
            'capm', 'capital asset pricing model', 'beta', 'alpha', 'volatility',
            'sharpe ratio', 'portfolio', 'diversification', 'risk premium',
            'efficient frontier', 'var', 'value at risk', 'options', 'derivatives',
            'black-scholes', 'monte carlo', 'regression', 'correlation',
            'interest rate', 'bond', 'equity', 'dividend', 'yield'
        ]
        
    def generate_answer(self, question: str, retrieved_chunks: List[Dict[str, Any]], 
                       citations: List[Any]) -> RAGResponse:
        """
        Generate answer using enhanced financial RAG chain.
        
        Args:
            question: User question
            retrieved_chunks: Retrieved document chunks
            citations: Citation data for chunks
            
        Returns:
            RAGResponse with structured answer and metadata
        """
        logger.info("Generating answer using Financial RAG Chain")
        
        try:
            # Enhance context with citations
            enhanced_context = self._enhance_context_with_citations(retrieved_chunks, citations)
            
            # Create dynamic prompt
            user_prompt = self._create_user_prompt(question, enhanced_context)
            
            # Generate response using LLM
            raw_response = self._call_llm(user_prompt)
            
            # Parse and structure response
            structured_response = self._parse_response(raw_response, citations)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(retrieved_chunks, structured_response, question)
            
            return RAGResponse(
                answer=structured_response.get('answer', ''),
                explanation=structured_response.get('explanation', ''),
                sources=structured_response.get('sources', []),
                confidence=confidence,
                raw_context=enhanced_context
            )
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return self._create_error_response(str(e))
    
    def _enhance_context_with_citations(self, chunks: List[Dict[str, Any]], 
                                      citations: List[Any]) -> str:
        """
        Enhance retrieved context with citation information.
        
        Args:
            chunks: Retrieved chunks
            citations: Citation objects
            
        Returns:
            Enhanced context string with citations
        """
        enhanced_context_parts = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.get('content', '')
            citation = citations[i] if i < len(citations) and citations[i] else None
            
            if citation:
                # Add citation prefix to chunk
                citation_prefix = f"[Source: {citation.document_title}, p.{citation.page_number}"
                if citation.section_title:
                    citation_prefix += f", {citation.section_title}"
                citation_prefix += "]"
                
                enhanced_chunk = f"{citation_prefix}\n{content}"
            else:
                enhanced_chunk = content
            
            enhanced_context_parts.append(enhanced_chunk)
        
        return "\n\n".join(enhanced_context_parts)
    
    def _create_user_prompt(self, question: str, context: str) -> str:
        """
        Create the user prompt with context and few-shot example.
        
        Args:
            question: User question
            context: Enhanced context with citations
            
        Returns:
            Complete user prompt
        """
        prompt = f"""Answer the following question about financial research based on the provided context.

Context:
{context}

Question: {question}

{self.few_shot_example}

Now provide your answer following the same format, ensuring accuracy and proper citations:
"""
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the language model with the constructed prompt.
        
        Args:
            prompt: Complete prompt for the LLM
            
        Returns:
            Raw response from LLM
        """
        if not self.llm_client:
            # Fallback response for testing
            return """Answer: Based on the provided context, I can analyze the financial information.

Explanation: The analysis requires examination of the specific data and methodologies presented in the sources.

Sources:
1. [Document Title, Page X, Section Y] - Relevant excerpt from the source.

Confidence: Medium - Answer based on available information with standard financial methodology."""
        
        try:
            # Construct messages for chat-based models
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",  # or specified model
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise
    
    def _parse_response(self, raw_response: str, citations: List[Any]) -> Dict[str, Any]:
        """
        Parse LLM response into structured components.
        
        Args:
            raw_response: Raw response from LLM
            citations: Available citations
            
        Returns:
            Structured response dictionary
        """
        # Initialize response structure
        structured = {
            'answer': '',
            'explanation': '',
            'sources': [],
            'confidence_text': ''
        }
        
        # Parse answer section
        answer_match = re.search(r'Answer:\s*(.*?)(?=\n\nExplanation:|$)', raw_response, re.DOTALL)
        if answer_match:
            structured['answer'] = answer_match.group(1).strip()
        
        # Parse explanation section
        explanation_match = re.search(r'Explanation:\s*(.*?)(?=\n\nSources?:|$)', raw_response, re.DOTALL)
        if explanation_match:
            structured['explanation'] = explanation_match.group(1).strip()
        
        # Parse sources section
        sources_match = re.search(r'Sources?:\s*(.*?)(?=\n\nConfidence:|$)', raw_response, re.DOTALL)
        if sources_match:
            sources_text = sources_match.group(1).strip()
            structured['sources'] = self._parse_sources(sources_text, citations)
        
        # Parse confidence section
        confidence_match = re.search(r'Confidence:\s*(.*?)$', raw_response, re.DOTALL)
        if confidence_match:
            structured['confidence_text'] = confidence_match.group(1).strip()
        
        return structured
    
    def _parse_sources(self, sources_text: str, citations: List[Any]) -> List[Dict[str, Any]]:
        """
        Parse sources from response text.
        
        Args:
            sources_text: Raw sources text
            citations: Available citations
            
        Returns:
            List of parsed source information
        """
        sources = []
        
        # Split by numbered items
        source_items = re.split(r'\n\s*\d+\.', sources_text)
        
        for item in source_items:
            item = item.strip()
            if not item:
                continue
            
            # Extract citation and quote
            citation_match = re.search(r'\[(.*?)\]', item)
            if citation_match:
                citation_text = citation_match.group(1)
                
                # Extract quote if present
                quote_match = re.search(r'-\s*["\']?(.*?)["\']?\s*$', item)
                quote = quote_match.group(1).strip() if quote_match else ""
                
                sources.append({
                    'citation': citation_text,
                    'quote': quote,
                    'full_text': item
                })
        
        return sources
    
    def _calculate_confidence_score(self, chunks: List[Dict[str, Any]], 
                                  response: Dict[str, Any], question: str) -> Dict[str, Any]:
        """
        Calculate confidence score based on multiple factors.
        
        Args:
            chunks: Retrieved chunks
            response: Structured response
            question: Original question
            
        Returns:
            Confidence assessment dictionary
        """
        # Initialize confidence factors
        source_quality = self._assess_source_quality(chunks)
        source_consistency = self._assess_source_consistency(chunks)
        response_coherence = self._assess_response_coherence(response)
        domain_relevance = self._assess_domain_relevance(question, response)
        
        # Calculate weighted confidence score
        weights = {
            'source_quality': 0.3,
            'source_consistency': 0.3,
            'response_coherence': 0.2,
            'domain_relevance': 0.2
        }
        
        final_score = (
            weights['source_quality'] * source_quality +
            weights['source_consistency'] * source_consistency +
            weights['response_coherence'] * response_coherence +
            weights['domain_relevance'] * domain_relevance
        )
        
        # Determine confidence level
        if final_score > 0.8:
            level = "High"
            explanation = "Multiple high-quality sources provide consistent information with strong domain relevance."
        elif final_score > 0.6:
            level = "Medium"
            explanation = "Sources provide good information but may have some gaps or inconsistencies."
        elif final_score > 0.4:
            level = "Low"
            explanation = "Limited or conflicting information in available sources."
        else:
            level = "Very Low"
            explanation = "Insufficient or unreliable information to support the answer."
        
        return {
            'level': level,
            'score': final_score,
            'explanation': explanation,
            'factors': {
                'source_quality': source_quality,
                'source_consistency': source_consistency,
                'response_coherence': response_coherence,
                'domain_relevance': domain_relevance
            }
        }
    
    def _assess_source_quality(self, chunks: List[Dict[str, Any]]) -> float:
        """Assess the quality of source chunks."""
        if not chunks:
            return 0.0
        
        quality_score = 0.0
        
        for chunk in chunks:
            chunk_score = 0.0
            
            # Check chunk length (not too short, not too long)
            content_length = len(chunk.get('content', ''))
            if 100 <= content_length <= 2000:
                chunk_score += 0.3
            
            # Check for mathematical content (important for financial papers)
            if chunk.get('formula_count', 0) > 0:
                chunk_score += 0.2
            
            # Check for tables (valuable data sources)
            if chunk.get('table_count', 0) > 0:
                chunk_score += 0.2
            
            # Check section information
            if chunk.get('section_id'):
                chunk_score += 0.3
            
            quality_score += min(chunk_score, 1.0)
        
        return quality_score / len(chunks)
    
    def _assess_source_consistency(self, chunks: List[Dict[str, Any]]) -> float:
        """Assess consistency between source chunks."""
        if len(chunks) < 2:
            return 1.0  # Single source is consistent with itself
        
        # Check if chunks come from same document/section
        documents = set()
        sections = set()
        
        for chunk in chunks:
            if 'document_title' in chunk:
                documents.add(chunk['document_title'])
            if 'section_id' in chunk:
                sections.add(chunk['section_id'])
        
        # Higher consistency if from same document or related sections
        doc_consistency = 1.0 / len(documents) if documents else 0.5
        section_consistency = 1.0 / len(sections) if sections else 0.5
        
        return (doc_consistency + section_consistency) / 2
    
    def _assess_response_coherence(self, response: Dict[str, Any]) -> float:
        """Assess coherence of the generated response."""
        coherence_score = 0.0
        
        answer = response.get('answer', '')
        explanation = response.get('explanation', '')
        sources = response.get('sources', [])
        
        # Check if answer is present and reasonable length
        if answer and 50 <= len(answer) <= 500:
            coherence_score += 0.3
        
        # Check if explanation is present
        if explanation and len(explanation) > 100:
            coherence_score += 0.3
        
        # Check if sources are provided
        if sources and len(sources) > 0:
            coherence_score += 0.2
        
        # Check for financial terminology usage
        combined_text = (answer + " " + explanation).lower()
        financial_term_count = sum(1 for term in self.financial_terms if term in combined_text)
        if financial_term_count > 0:
            coherence_score += 0.2
        
        return min(coherence_score, 1.0)
    
    def _assess_domain_relevance(self, question: str, response: Dict[str, Any]) -> float:
        """Assess relevance to financial domain."""
        question_lower = question.lower()
        answer_lower = response.get('answer', '').lower()
        
        # Check if question contains financial terms
        question_financial = any(term in question_lower for term in self.financial_terms)
        
        # Check if answer contains financial terms
        answer_financial = any(term in answer_lower for term in self.financial_terms)
        
        # Check for mathematical content indicators
        has_numbers = bool(re.search(r'\d+\.?\d*%?', answer_lower))
        has_formulas = bool(re.search(r'[α-ωΑ-Ω]|formula|equation', answer_lower))
        
        relevance_score = 0.0
        if question_financial:
            relevance_score += 0.3
        if answer_financial:
            relevance_score += 0.3
        if has_numbers:
            relevance_score += 0.2
        if has_formulas:
            relevance_score += 0.2
        
        return min(relevance_score, 1.0)
    
    def _create_error_response(self, error_message: str) -> RAGResponse:
        """Create error response when generation fails."""
        return RAGResponse(
            answer="I apologize, but I encountered an error while processing your question.",
            explanation=f"Error details: {error_message}",
            sources=[],
            confidence={
                'level': 'Very Low',
                'score': 0.0,
                'explanation': 'Unable to generate response due to technical error.',
                'factors': {}
            },
            raw_context=""
        )