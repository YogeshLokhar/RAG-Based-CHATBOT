import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.indexes import SQLRecordManager, index

# Set up embeddings
@st.cache_resource
def get_embedding_function():
    return OllamaEmbeddings(model='nomic-embed-text')

# Load and process documents
@st.cache_resource
def load_and_process_documents():
    loader = DirectoryLoader('/home/computer/project/dataset/', glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=100)
    return text_splitter.split_documents(docs)

# Initialize and populate Chroma vector store
@st.cache_resource
def get_vectorstore():
    embedding_function = get_embedding_function()
    texts = load_and_process_documents()
    
    vectorstore = Chroma(persist_directory="/home/computer/project/database", embedding_function=embedding_function)
    
    # Set up record manager for indexing
    collection_name = "pdf_documents"
    namespace = f"chroma/{collection_name}"
    record_manager = SQLRecordManager(namespace, db_url="sqlite:///record_manager_cache.sql")
    record_manager.create_schema()
    
    # Index the documents
    index_result = index(
        texts,
        record_manager,
        vectorstore,
        cleanup="incremental",
        source_id_key="source"
    )
    
    st.write(f"Indexing result: {index_result}")

# Streamlit UI for loading documents
st.title("Load and Index Documents")

# Initialize components
with st.spinner("Loading and indexing documents... This may take a while."):
    get_vectorstore()

st.success("Documents loaded and indexed successfully!")
