"""
Enhanced RAG system with Hybrid Search integration
"""
import logging
import streamlit as st
from typing import List, Dict, Any, Optional
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils.config import get_api_keys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRAG:
    """Enhanced RAG with hybrid search capabilities"""
    
    def __init__(self):
        openai_key, _, _ = get_api_keys()
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini", 
            temperature=0,
            max_tokens=1500,
            openai_api_key=openai_key
        )
        
        self.prompt_template = PromptTemplate.from_template(
            """Ответьте на вопрос на основе предоставленного контекста из научных и финансовых документов.

            КОНТЕКСТ:
            {context}

            ВОПРОС: {question}

            ИНСТРУКЦИИ:
            - Отвечайте ТОЛЬКО на русском языке
            - Используйте ТОЛЬКО информацию из предоставленного контекста
            - Если в контексте есть математические формулы (например, ∆c = e^(-qt_opt) · N(d1)), воспроизводите их точно
            - Объясняйте финансовые и математические термины простым языком
            - Если контекст содержит формулы или таблицы, обязательно включите их в ответ
            - Укажите источники (номера документов), если доступны
            - Если ответ отсутствует в контексте, честно сообщите: "В предоставленном контексте нет информации для ответа на этот вопрос"
            - Форматируйте ответ для лучшей читаемости с использованием списков и подзаголовков

            СПЕЦИАЛЬНЫЕ ПРАВИЛА:
            - Delta (Δ) - чувствительность цены опциона к изменению цены базового актива
            - Vega (V) - чувствительность цены опциона к изменению волатильности
            - Сохраняйте математические обозначения и греческие буквы
            - Объясняйте значение переменных в формулах

            ОТВЕТ:"""
        )
    
    def query_with_vectorstore(self, question: str, vectorstore) -> str:
        """Query using traditional vectorstore (fallback method)"""
        logger.info(f"Using vectorstore query for: {question[:100]}...")
        
        try:
            from utils.rag import ask_question
            return ask_question(question, vectorstore)
        except Exception as e:
            logger.error(f"Vectorstore query failed: {e}")
            return f"❌ Ошибка при поиске: {str(e)}"
    
    def query_with_enhanced_processor(self, question: str, enhanced_processor) -> str:
        """Query using EnhancedDocumentProcessor with hybrid search"""
        logger.info(f"Using enhanced processor query for: {question[:100]}...")
        
        try:
            # Get current index
            current_index = st.session_state.current_index
            if not current_index:
                return "❌ Индекс не выбран"
            
            # Use enhanced processor query
            result = enhanced_processor.query_documents(
                question=question,
                index_name=current_index,
                top_k=5
            )
            
            if result['status'] == 'success':
                return result['formatted_response']
            else:
                return f"❌ Ошибка обработки: {result.get('error_message', 'Unknown error')}"
                
        except Exception as e:
            logger.error(f"Enhanced processor query failed: {e}")
            return f"❌ Ошибка при поиске: {str(e)}"
    
    def query_hybrid(self, question: str, vectorstore=None, enhanced_processor=None) -> str:
        """
        Hybrid query that tries enhanced processor first, falls back to vectorstore
        """
        logger.info(f"Starting hybrid query for: {question[:100]}...")
        
        # Try enhanced processor first if available
        if enhanced_processor:
            try:
                logger.info("Attempting enhanced processor query...")
                result = self.query_with_enhanced_processor(question, enhanced_processor)
                
                # Check if result is meaningful (not an error or "no information" response)
                if not result.startswith("❌") and "нет информации" not in result.lower():
                    logger.info("Enhanced processor query succeeded")
                    return result
                else:
                    logger.warning("Enhanced processor returned insufficient result, trying fallback")
            except Exception as e:
                logger.warning(f"Enhanced processor failed: {e}, trying fallback")
        
        # Fallback to vectorstore
        if vectorstore:
            logger.info("Using vectorstore fallback...")
            return self.query_with_vectorstore(question, vectorstore)
        
        return "❌ Нет доступных методов поиска"

# Global instance
enhanced_rag = EnhancedRAG()

def ask_question_enhanced(question: str, vectorstore=None, enhanced_processor=None) -> str:
    """
    Enhanced question answering with hybrid approach
    
    Args:
        question: User question
        vectorstore: Traditional vectorstore (fallback)
        enhanced_processor: EnhancedDocumentProcessor (preferred)
    
    Returns:
        Answer string
    """
    return enhanced_rag.query_hybrid(question, vectorstore, enhanced_processor)