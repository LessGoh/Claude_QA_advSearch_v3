import streamlit as st
import uuid
from datetime import datetime
from utils.rag import ask_question
from utils.enhanced_rag import ask_question_enhanced
from utils.chat_history import ChatHistoryManager
from utils.auth import get_user_index_id
from utils.config import get_api_keys
import time

class ChatPage:
    def __init__(self):
        self.chat_manager = ChatHistoryManager()
        
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
        st.markdown("# 💬 Чат с документами")
        
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
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📊 **Текущий индекс:** {index_type}")
        with col2:
            stats = st.session_state.index_manager.get_index_stats(current_index_name)
            if stats["status"] == "success":
                st.info(f"📚 **Документов:** {stats['total_vectors']:,}")
            else:
                st.info("📚 **Документов:** Недоступно")
        with col3:
            # Show enhanced processor status
            if self.enhanced_processor:
                st.success("🚀 **v3 Поиск:** Активен")
            else:
                st.warning("⚠️ **v3 Поиск:** Отключен")
        
        # Initialize conversation
        if 'current_conversation_id' not in st.session_state:
            st.session_state.current_conversation_id = str(uuid.uuid4())
        
        # Load chat history
        chat_history = self.chat_manager.get_conversation_history(
            st.session_state.current_conversation_id,
            st.session_state.username
        )
        
        # Display chat history
        st.markdown("## 📜 История диалога")
        
        chat_container = st.container()
        with chat_container:
            if chat_history:
                for i, message in enumerate(chat_history):
                    self._render_message(message, i)
            else:
                st.info("👋 Начните диалог, задав вопрос о документах!")
        
        # Chat input
        st.markdown("---")
        st.markdown("## ❓ Задать вопрос")
        
        with st.form("chat_form", clear_on_submit=True):
            question = st.text_area(
                "Ваш вопрос:",
                placeholder="Что вы хотели бы узнать из документов?",
                height=100,
                key="question_input"
            )
            
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                submit_button = st.form_submit_button("🚀 Отправить")
            with col2:
                clear_button = st.form_submit_button("🗑️ Очистить")
        
        # Handle form submission
        if submit_button and question.strip():
            self._handle_question(question.strip(), vectorstore)
        elif clear_button:
            self._clear_conversation()
    
    def _render_message(self, message, index):
        """Render a single message in the chat"""
        timestamp = datetime.fromisoformat(message['timestamp']).strftime("%H:%M")
        
        if message['type'] == 'question':
            # User message
            st.markdown(f"""
            <div style="text-align: right; margin: 10px 0;">
                <div style="background-color: #007ACC; color: white; padding: 10px 15px; 
                           border-radius: 15px 15px 5px 15px; display: inline-block; max-width: 70%;">
                    <strong>Вы ({timestamp}):</strong><br>
                    {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Bot response
            st.markdown(f"""
            <div style="text-align: left; margin: 10px 0;">
                <div style="background-color: #f0f0f0; color: black; padding: 10px 15px; 
                           border-radius: 15px 15px 15px 5px; display: inline-block; max-width: 70%;">
                    <strong>🤖 Бот ({timestamp}):</strong><br>
                    {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def _handle_question(self, question, vectorstore):
        """Handle user question"""
        # Add question to history
        question_message = {
            'type': 'question',
            'content': question,
            'timestamp': datetime.now().isoformat()
        }
        
        self.chat_manager.add_message(
            st.session_state.current_conversation_id,
            st.session_state.username,
            question_message
        )
        
        # Show typing indicator
        with st.spinner("🤔 Ищу ответ в документах..."):
            time.sleep(0.5)  # Brief delay for UX
            
            # Use enhanced RAG if available
            if self.enhanced_processor:
                st.write("🚀 Используется улучшенный поиск v3...")
                answer = ask_question_enhanced(question, vectorstore, self.enhanced_processor)
            else:
                st.write("⚠️ Используется базовый поиск...")
                answer = ask_question(question, vectorstore)
        
        # Add answer to history
        answer_message = {
            'type': 'answer',
            'content': answer,
            'timestamp': datetime.now().isoformat()
        }
        
        self.chat_manager.add_message(
            st.session_state.current_conversation_id,
            st.session_state.username,
            answer_message
        )
        
        st.rerun()
    
    def _clear_conversation(self):
        """Clear current conversation"""
        self.chat_manager.clear_conversation(
            st.session_state.current_conversation_id,
            st.session_state.username
        )
        st.success("💬 История диалога очищена!")
        st.rerun()

def render_chat_page():
    """Render the chat page"""
    chat_page = ChatPage()
    chat_page.render()