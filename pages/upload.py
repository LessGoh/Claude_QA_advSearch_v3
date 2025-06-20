import streamlit as st
import tempfile
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from utils.document_manager import DocumentManager
from utils.auth import get_user_index_id
from utils.enhanced_document_processor import EnhancedDocumentProcessor
from utils.config import get_api_keys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class UploadPage:
    def __init__(self):
        self.doc_manager = DocumentManager()
        
        # Initialize enhanced processor
        openai_key, pinecone_key, _ = get_api_keys()
        if openai_key and pinecone_key:
            self.enhanced_processor = EnhancedDocumentProcessor(openai_key, pinecone_key)
        else:
            self.enhanced_processor = None
    
    def render(self):
        st.markdown("# 📎 Загрузка документов")
        
        # Check if enhanced processor is available
        if not self.enhanced_processor:
            st.error("🔧 Система обработки не инициализирована. Проверьте настройки API ключей.")
            return
        
        # Check if vectorstore is available
        if not st.session_state.index_manager:
            st.error("Менеджер индексов не инициализирован")
            return
        
        # Current index info
        user_id = get_user_index_id(st.session_state.username)
        current_index_name = st.session_state.current_index or f"pdf-qa-personal-{user_id}"
        index_type = "Личный" if current_index_name.endswith(f"-{user_id}") else "Общий"
        
        st.info(f"📊 **Загрузка в индекс:** {index_type}")
        
        # Index selection
        st.markdown("## 🎯 Выбор индекса для загрузки")
        
        col1, col2 = st.columns(2)
        with col1:
            target_index = st.radio(
                "Куда загрузить документы:",
                ["Личный индекс", "Общий индекс"],
                index=0 if index_type == "Личный" else 1,
                help="Личный индекс - только для вас, Общий - для всей команды"
            )
        
        with col2:
            if target_index == "Личный индекс":
                st.info("🔒 Документы будут доступны только вам")
            else:
                st.info("👥 Документы будут доступны всей команде")
        
        # File upload section
        st.markdown("## 📁 Загрузка файлов")
        
        # Enhanced features info
        st.info("🚀 **Новые возможности v3:**\n"
                "• Интеллектуальная детекция структуры документа\n"
                "• Сохранение математических формул и таблиц\n"
                "• Семантическое группирование текста\n"
                "• Точные ссылки с номерами страниц")
        
        # Drag and drop area
        uploaded_files = st.file_uploader(
            "Перетащите файлы сюда или нажмите для выбора",
            type="pdf",
            accept_multiple_files=True,
            help="Поддерживаются только PDF файлы. Максимальный размер: 50MB на файл"
        )
        
        if uploaded_files:
            self._display_file_list(uploaded_files)
            
            # Process button
            col1, col2 = st.columns([3, 1])
            with col1:
                use_enhanced = st.checkbox("🔬 Использовать улучшенную обработку v3", 
                                         value=True,
                                         help="Включает детекцию структуры, умное разбиение и точные цитаты")
            with col2:
                if st.button("🚀 Загрузить документы", use_container_width=True):
                    self._process_files(uploaded_files, target_index, use_enhanced)
    
    def _display_file_list(self, uploaded_files):
        """Display list of uploaded files with validation"""
        st.markdown("### 📋 Выбранные файлы:")
        
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        valid_files = []
        invalid_files = []
        
        for i, file in enumerate(uploaded_files):
            file_size_mb = file.size / (1024 * 1024)
            
            if file.size > MAX_FILE_SIZE:
                invalid_files.append(f"{file.name} ({file_size_mb:.1f}MB - слишком большой)")
                st.error(f"❌ {file.name} ({file_size_mb:.1f}MB) - превышен лимит размера")
            else:
                valid_files.append(file)
                st.success(f"✅ {file.name} ({file_size_mb:.1f}MB)")
        
        if invalid_files:
            st.warning("⚠️ Некоторые файлы не будут обработаны из-за превышения лимита размера (50MB)")
        
        return valid_files
    
    def _process_files(self, uploaded_files, target_index, use_enhanced=True):
        """Process uploaded files with optional enhanced processing"""
        # Filter valid files
        MAX_FILE_SIZE = 50 * 1024 * 1024
        valid_files = [f for f in uploaded_files if f.size <= MAX_FILE_SIZE]
        
        if not valid_files:
            st.error("Нет файлов для обработки")
            return
        
        # Get appropriate vectorstore
        if target_index == "Личный индекс":
            vectorstore = st.session_state.index_manager.get_vectorstore("personal", st.session_state.username)
            user_id = get_user_index_id(st.session_state.username)
            index_name = f"pdf-qa-personal-{user_id}"
        else:
            vectorstore = st.session_state.index_manager.get_vectorstore("shared")
            index_name = "pdf-qa-shared"
        
        if not vectorstore:
            st.error("Не удалось подключиться к векторной базе данных")
            return
        
        # Process files
        if use_enhanced and self.enhanced_processor:
            results = self._process_multiple_pdfs_enhanced(valid_files, index_name)
        else:
            results = self._process_multiple_pdfs_legacy(valid_files, vectorstore, index_name)
        self._display_results(results)
    
    def _process_multiple_pdfs_enhanced(self, uploaded_files, index_name):
        """Process multiple PDF files with enhanced v3 processing"""
        def process_single_file_enhanced(file_data):
            file, index_name, username = file_data
            start_time = time.time()
            
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Process with enhanced processor
                result = self.enhanced_processor.process_document(
                    tmp_file_path, 
                    index_name,
                    document_id=None
                )
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
                # Store enhanced document metadata
                if result['status'] == 'success':
                    doc_metadata = {
                        'filename': file.name,
                        'upload_user': username,
                        'upload_date': datetime.now().isoformat(),
                        'file_size': file.size,
                        'chunk_count': result['chunks_created'],
                        'sections_detected': result['sections_detected'],
                        'has_formulas': result['has_formulas'],
                        'has_tables': result['has_tables'],
                        'index_name': index_name,
                        'status': 'success',
                        'document_id': result['document_id'],
                        'document_title': result['document_title'],
                        'processing_method': 'enhanced_v3'
                    }
                    
                    # Save to document manager
                    self.doc_manager.add_document(doc_metadata)
                
                processing_time = time.time() - start_time
                return {
                    "file": file.name,
                    "status": result['status'], 
                    "chunks": result.get('chunks_created', 0),
                    "sections": result.get('sections_detected', 0),
                    "has_formulas": result.get('has_formulas', False),
                    "has_tables": result.get('has_tables', False),
                    "document_title": result.get('document_title', file.name),
                    "processing_time": processing_time,
                    "processing_method": "enhanced_v3"
                }
                
            except Exception as e:
                # Clean up temporary file if it exists
                try:
                    if 'tmp_file_path' in locals():
                        os.unlink(tmp_file_path)
                except:
                    pass
                
                processing_time = time.time() - start_time
                return {
                    "file": file.name,
                    "status": "error",
                    "error": str(e),
                    "chunks": 0,
                    "processing_time": processing_time,
                    "processing_method": "enhanced_v3"
                }
        
        # Process files with progress tracking
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=2) as executor:  # Reduced workers for enhanced processing
            # Prepare file data for processing
            file_data = [(file, index_name, st.session_state.username) for file in uploaded_files]
            
            # Submit all tasks
            future_to_file = {
                executor.submit(process_single_file_enhanced, data): data[0].name 
                for data in file_data
            }
            
            # Process completed tasks
            completed = 0
            total_files = len(uploaded_files)
            
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                completed += 1
                
                # Update progress
                progress = completed / total_files
                progress_bar.progress(progress)
                status_text.text(f"Улучшенная обработка {filename}... ({completed}/{total_files})")
                
                # Get result
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "file": filename,
                        "status": "error",
                        "error": str(e),
                        "chunks": 0,
                        "processing_method": "enhanced_v3"
                    })
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return results
    
    def _process_multiple_pdfs_legacy(self, uploaded_files, vectorstore, index_name):
        """Process multiple PDF files with legacy processing"""
        def process_single_file(file_data):
            file, vectorstore, index_name, username = file_data
            start_time = time.time()
            
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Load and process PDF
                pdf_loader = PyPDFLoader(tmp_file_path)
                pages = pdf_loader.load_and_split()
                
                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=100
                )
                splits = text_splitter.split_documents(pages)
                
                # Add metadata
                for split in splits:
                    split.metadata.update({
                        'filename': file.name,
                        'upload_user': username,
                        'upload_date': datetime.now().isoformat(),
                        'file_size': file.size,
                        'index_name': index_name
                    })
                
                # Add documents to vector store
                vectorstore.add_documents(documents=splits)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
                # Store document metadata
                doc_metadata = {
                    'filename': file.name,
                    'upload_user': username,
                    'upload_date': datetime.now().isoformat(),
                    'file_size': file.size,
                    'chunk_count': len(splits),
                    'index_name': index_name,
                    'status': 'success'
                }
                
                # Save to document manager
                self.doc_manager.add_document(doc_metadata)
                
                processing_time = time.time() - start_time
                return {
                    "file": file.name,
                    "status": "success", 
                    "chunks": len(splits),
                    "processing_time": processing_time,
                    "processing_method": "legacy"
                }
                
            except Exception as e:
                # Clean up temporary file if it exists
                try:
                    if 'tmp_file_path' in locals():
                        os.unlink(tmp_file_path)
                except:
                    pass
                
                processing_time = time.time() - start_time
                return {
                    "file": file.name,
                    "status": "error",
                    "error": str(e),
                    "chunks": 0,
                    "processing_time": processing_time,
                    "processing_method": "legacy"
                }
        
        # Process files with progress tracking
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Prepare file data for processing
            file_data = [(file, vectorstore, index_name, st.session_state.username) for file in uploaded_files]
            
            # Submit all tasks
            future_to_file = {
                executor.submit(process_single_file, data): data[0].name 
                for data in file_data
            }
            
            # Process completed tasks
            completed = 0
            total_files = len(uploaded_files)
            
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                completed += 1
                
                # Update progress
                progress = completed / total_files
                progress_bar.progress(progress)
                status_text.text(f"Обработка {filename}... ({completed}/{total_files})")
                
                # Get result
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "file": filename,
                        "status": "error",
                        "error": str(e),
                        "chunks": 0
                    })
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return results
    
    def _display_results(self, results):
        """Display processing results with enhanced information"""
        successful_files = [r for r in results if r["status"] == "success"]
        failed_files = [r for r in results if r["status"] == "error"]
        enhanced_files = [r for r in successful_files if r.get("processing_method") == "enhanced_v3"]
        
        if successful_files:
            total_chunks = sum(r["chunks"] for r in successful_files)
            total_time = sum(r["processing_time"] for r in successful_files)
            
            # Enhanced statistics
            if enhanced_files:
                total_sections = sum(r.get("sections", 0) for r in enhanced_files)
                formula_docs = len([r for r in enhanced_files if r.get("has_formulas", False)])
                table_docs = len([r for r in enhanced_files if r.get("has_tables", False)])
                
                st.success(f"✅ {len(successful_files)} документов успешно загружено! "
                          f"Создано {total_chunks} фрагментов за {total_time:.1f}с.")
                
                if enhanced_files:
                    st.info(f"🔬 **Расширенная обработка (v3):** {len(enhanced_files)} документов\n"
                           f"• Обнаружено разделов: {total_sections}\n"
                           f"• Документов с формулами: {formula_docs}\n"
                           f"• Документов с таблицами: {table_docs}")
            else:
                st.success(f"✅ {len(successful_files)} документов успешно загружено! Создано {total_chunks} фрагментов за {total_time:.1f}с.")
            
            # Show successful files with detailed info
            with st.expander("📋 Успешно обработанные файлы"):
                for result in successful_files:
                    method = result.get("processing_method", "legacy")
                    
                    if method == "enhanced_v3":
                        # Enhanced processing result
                        extras = []
                        if result.get("sections", 0) > 0:
                            extras.append(f"{result['sections']} разделов")
                        if result.get("has_formulas", False):
                            extras.append("формулы")
                        if result.get("has_tables", False):
                            extras.append("таблицы")
                        
                        extra_info = f" ({', '.join(extras)})" if extras else ""
                        title = result.get("document_title", result["file"])
                        
                        st.write(f"✅🔬 **{result['file']}** - {title}")
                        st.write(f"    └ {result['chunks']} фрагментов{extra_info} ({result['processing_time']:.1f}с)")
                    else:
                        # Legacy processing result
                        st.write(f"✅ {result['file']} - {result['chunks']} фрагментов ({result['processing_time']:.1f}с)")
        
        if failed_files:
            st.error(f"❌ {len(failed_files)} документов не удалось обработать")
            # Show failed files
            with st.expander("⚠️ Ошибки обработки"):
                for result in failed_files:
                    method = result.get("processing_method", "unknown")
                    st.write(f"❌ {result['file']} ({method}) - {result['error']}")

def render_upload_page():
    """Render the upload page"""
    upload_page = UploadPage()
    upload_page.render()