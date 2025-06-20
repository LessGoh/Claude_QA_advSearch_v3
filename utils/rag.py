from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import logging
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_docs(docs):
    """Format retrieved documents with v3 metadata and detailed logging"""
    logger.info(f"Formatting {len(docs)} retrieved documents")
    
    if not docs:
        logger.warning("No documents retrieved!")
        return "Документы не найдены в базе данных."
    
    formatted_content = []
    for i, doc in enumerate(docs):
        logger.info(f"Document {i+1}: {len(doc.page_content)} characters")
        logger.info(f"Document {i+1} metadata: {doc.metadata}")
        
        # Extract v3 metadata if available
        metadata = doc.metadata or {}
        
        # Build document header with enhanced metadata
        doc_header = f"[Документ {i+1}"
        
        # Add document title if available
        if 'document_title' in metadata:
            doc_header += f" - {metadata['document_title']}"
        elif 'filename' in metadata:
            doc_header += f" - {metadata['filename']}"
        
        # Add page number if available
        if 'page_num' in metadata:
            doc_header += f", стр. {metadata['page_num']}"
        
        # Add section info if available  
        if 'section_title' in metadata:
            doc_header += f", раздел: {metadata['section_title']}"
        
        # Add content type indicators
        indicators = []
        if metadata.get('formula_count', 0) > 0:
            indicators.append(f"📐 формул: {metadata['formula_count']}")
        if metadata.get('table_count', 0) > 0:
            indicators.append(f"📊 таблиц: {metadata['table_count']}")
        if metadata.get('content_type') == 'mathematical':
            indicators.append("🔢 математический контент")
        
        if indicators:
            doc_header += f" ({', '.join(indicators)})"
        
        doc_header += "]"
        
        # Format the content
        content = doc.page_content
        
        # Highlight mathematical formulas if they exist
        if metadata.get('formula_count', 0) > 0:
            content = content.replace('∆', 'Δ')  # Normalize Delta symbols
            content = content.replace('δ', 'Δ')  # Normalize delta symbols
        
        formatted_content.append(f"{doc_header}\n{content}")
    
    result = "\n\n".join(formatted_content)
    logger.info(f"Total formatted content length: {len(result)} characters")
    
    # Count mathematical content
    math_indicators = sum(1 for doc in docs if doc.metadata.get('formula_count', 0) > 0)
    if math_indicators > 0:
        logger.info(f"Found {math_indicators} documents with mathematical formulas")
    
    return result

def ask_question(question, vectorstore):
    """Get answer to question using RAG with detailed logging"""
    logger.info(f"Processing question: {question[:100]}...")
    
    try:
        # Check vectorstore status
        logger.info(f"Vectorstore type: {type(vectorstore)}")
        
        # Try to get index stats if available
        try:
            if hasattr(vectorstore, '_index') and hasattr(vectorstore._index, 'describe_index_stats'):
                stats = vectorstore._index.describe_index_stats()
                logger.info(f"Index stats: {stats}")
        except Exception as e:
            logger.warning(f"Could not get index stats: {e}")
        
        # Enhanced search parameters for better mathematical content retrieval
        search_kwargs = {"k": 5}
        
        # Check if question contains mathematical terms
        math_terms = ['delta', 'дельта', 'vega', 'вега', 'формула', 'formula', 'опцион', 'option', 
                     'волатильность', 'volatility', '∆', 'Δ', 'греки', 'greeks']
        
        question_lower = question.lower()
        has_math_terms = any(term in question_lower for term in math_terms)
        
        if has_math_terms:
            logger.info("Mathematical terms detected in question, adjusting search parameters")
            # Increase retrieval count for mathematical queries
            search_kwargs["k"] = 8
            
            # Add filter for mathematical content if available
            search_kwargs["filter"] = {
                "formula_count": {"$gte": 0}  # Include documents with formulas
            }
        
        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        
        # Test retrieval directly
        logger.info("Testing document retrieval...")
        try:
            retrieved_docs = retriever.get_relevant_documents(question)
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            # Log each retrieved document
            for i, doc in enumerate(retrieved_docs):
                logger.info(f"Retrieved doc {i+1}: content length={len(doc.page_content)}, metadata={doc.metadata}")
                logger.info(f"Retrieved doc {i+1} content preview: {doc.page_content[:200]}...")
                
        except Exception as e:
            logger.error(f"Direct retrieval failed: {e}")
            return f"❌ Ошибка при поиске документов: {str(e)}"
        
        prompt_rag = PromptTemplate.from_template(
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
        
        llm = ChatOpenAI(
            model_name="gpt-4o-mini", 
            temperature=0,
            max_tokens=1000
        )
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_rag
            | llm
        )
        
        # Log the chain execution
        logger.info("Executing RAG chain...")
        result = rag_chain.invoke(question)
        
        logger.info(f"Generated response length: {len(result.content)} characters")
        logger.info(f"Response preview: {result.content[:200]}...")
        
        # Display debug info in Streamlit for troubleshooting
        if st.session_state.get('debug_mode', False):
            with st.expander("🔍 Debug Information"):
                st.write(f"**Question:** {question}")
                st.write(f"**Retrieved documents:** {len(retrieved_docs)}")
                for i, doc in enumerate(retrieved_docs):
                    st.write(f"**Document {i+1}:** {len(doc.page_content)} chars")
                    st.code(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                st.write(f"**Generated response:** {len(result.content)} chars")
        
        return result.content
    
    except Exception as e:
        error_msg = f"❌ Ошибка при обработке вопроса: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full exception details:")
        return error_msg