import streamlit as st
from utils.rag import ask_question
from utils.enhanced_rag import ask_question_enhanced
from utils.auth import get_user_index_id
from utils.config import get_api_keys
import logging

class DebugPage:
    def __init__(self):
        # Initialize enhanced processor
        openai_key, pinecone_key, _ = get_api_keys()
        if openai_key and pinecone_key:
            try:
                from utils.enhanced_document_processor import EnhancedDocumentProcessor
                self.enhanced_processor = EnhancedDocumentProcessor(openai_key, pinecone_key)
            except Exception as e:
                st.warning(f"Не удалось инициализировать улучшенный процессор: {e}")
                self.enhanced_processor = None
        else:
            self.enhanced_processor = None
    
    def render(self):
        st.markdown("# 🔍 Отладка поиска")
        
        # Enable debug mode
        st.session_state.debug_mode = True
        
        # Check if vectorstore is available
        if not st.session_state.index_manager:
            st.error("Менеджер индексов не инициализирован")
            return
        
        vectorstore = st.session_state.index_manager.get_current_vectorstore()
        if not vectorstore:
            st.error("Не удалось подключиться к векторной базе данных")
            return
        
        # Display current index info
        current_index_name = st.session_state.current_index or "Не выбран"
        user_id = get_user_index_id(st.session_state.username)
        index_type = "Личный" if current_index_name.endswith(f"-{user_id}") else "Общий"
        
        st.info(f"📊 **Текущий индекс:** {index_type} ({current_index_name})")
        
        # Index statistics
        try:
            stats = st.session_state.index_manager.get_index_stats(current_index_name)
            if stats["status"] == "success":
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Векторов в индексе", stats['total_vectors'])
                with col2:
                    st.metric("Размерность", stats.get('dimension', 'N/A'))
                with col3:
                    fullness = stats.get('index_fullness', 0) * 100
                    st.metric("Заполненность", f"{fullness:.1f}%")
        except Exception as e:
            st.warning(f"Не удалось получить статистику индекса: {e}")
        
        # Test search functionality
        st.markdown("## 🔍 Тестирование поиска")
        
        # Expected content info
        st.info("""
        **Ожидаемый контент в вашем документе:**
        - Delta measures an option's price sensitivity to changes in the price of the underlying asset
        - ∆c = e−qtopt · N(d1) для call опционов  
        - ∆p = −e−qtopt · N(−d1) для put опционов
        - Vega measures the sensitivity of the option price to changes in implied volatility
        - V = P0√topte^(-d₁²/2) √(1/2π)
        - Daily management of Delta exposure and regular adjustments to Vega exposure
        """)
        
        # Debug mode toggle
        debug_mode = st.checkbox("🔍 Режим отладки (показать детали поиска)", value=True)
        st.session_state.debug_mode = debug_mode
        
        # Predefined test questions
        test_questions = [
            "Delta", 
            "формула Delta",
            "что такое Delta опциона?",
            "vega измеряет чувствительность",
            "опцион",
            "volatility",
            "∆c = e−qtopt · N(d1)",
            "вега чувствительность",
            "математические формулы",
            "Daily management of Delta exposure",
            "число опционов за временной шаг"
        ]
        
        col1, col2 = st.columns([2, 1])
        with col1:
            question = st.text_input(
                "Введите вопрос для тестирования:",
                placeholder="Например: что такое Delta?"
            )
        with col2:
            selected_test = st.selectbox("Или выберите тестовый вопрос:", [""] + test_questions)
        
        if selected_test:
            question = selected_test
        
        # Search method selection
        search_method = st.radio(
            "Выберите метод поиска:",
            ["🚀 Улучшенный v3 (рекомендуется)", "⚠️ Базовый векторный поиск", "🔄 Гибридный (v3 + fallback)"],
            index=0 if self.enhanced_processor else 1
        )
        
        if st.button("🚀 Протестировать поиск", disabled=not question):
            if question:
                self._test_search(question, vectorstore, search_method)
        
        # Direct index query test
        st.markdown("## 🔬 Прямой запрос к индексу")
        
        if st.button("📊 Показать образцы векторов"):
            self._show_sample_vectors(vectorstore)
    
    def _test_search(self, question, vectorstore, search_method):
        """Test search with detailed output"""
        st.markdown(f"### Тестирование вопроса: '{question}'")
        st.markdown(f"**Метод поиска:** {search_method}")
        
        # Enable logging display
        import io
        import sys
        
        # Capture logs
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger = logging.getLogger('utils.rag')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        try:
            # Test the search based on selected method
            with st.spinner("Выполняется поиск..."):
                if search_method.startswith("🚀") and self.enhanced_processor:
                    # Enhanced v3 search
                    from utils.enhanced_rag import enhanced_rag
                    answer = enhanced_rag.query_with_enhanced_processor(question, self.enhanced_processor)
                elif search_method.startswith("🔄"):
                    # Hybrid search
                    answer = ask_question_enhanced(question, vectorstore, self.enhanced_processor)
                else:
                    # Basic vectorstore search
                    answer = ask_question(question, vectorstore)
            
            # Show results
            st.markdown("#### 💬 Ответ:")
            st.write(answer)
            
            # Show captured logs
            log_output = log_capture.getvalue()
            if log_output:
                st.markdown("#### 📋 Логи поиска:")
                st.code(log_output, language="text")
            
        except Exception as e:
            st.error(f"Ошибка при тестировании: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            logger.removeHandler(handler)
    
    def _show_sample_vectors(self, vectorstore):
        """Show sample vectors from the index"""
        st.markdown("### 📊 Образцы векторов в индексе")
        
        try:
            # Try to get some sample data
            if hasattr(vectorstore, '_index'):
                index = vectorstore._index
                
                # Try to query with a dummy vector to see what's stored
                dummy_query = [0.0] * 1536  # OpenAI embedding dimension
                
                query_result = index.query(
                    vector=dummy_query,
                    top_k=5,
                    include_metadata=True
                )
                
                st.write(f"Найдено {len(query_result.matches)} векторов в индексе:")
                
                for i, match in enumerate(query_result.matches):
                    with st.expander(f"Вектор {i+1} (ID: {match.id})"):
                        st.write(f"**Score:** {match.score}")
                        st.write("**Метаданные:**")
                        st.json(match.metadata)
                        
                        # Show content if available
                        if 'content' in match.metadata:
                            st.write("**Контент:**")
                            content = match.metadata['content']
                            st.code(content[:500] + "..." if len(content) > 500 else content)
            else:
                st.warning("Не удалось получить доступ к индексу для просмотра образцов")
                
        except Exception as e:
            st.error(f"Ошибка при получении образцов: {e}")
            import traceback
            st.code(traceback.format_exc())

def render_debug_page():
    """Render the debug page"""
    debug_page = DebugPage()
    debug_page.render()