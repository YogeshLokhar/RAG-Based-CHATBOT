import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
import csv
from datetime import datetime
import threading
import time

# Ensure this function is either defined here or imported
def get_embedding_function():
    from langchain_community.embeddings import OllamaEmbeddings
    return OllamaEmbeddings(model='nomic-embed-text')

# Initialize Ollama LLM
@st.cache_resource
def get_llm():
    return Ollama(model="qwen2")

# Create a retrieval chain
@st.cache_resource
def get_qa_chain():
    embedding_function = get_embedding_function()
    vectorstore = Chroma(persist_directory="/home/computer/project/database", embedding_function=embedding_function)
    llm = get_llm()
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

# Function to calculate metrics and log to CSV
def calculate_and_log_metrics(query, result, start_time, end_time, load_start_time, load_end_time, prompt_eval_start_time, prompt_eval_end_time):
    total_duration = (end_time - start_time) * 1e9  # Convert to nanoseconds
    load_duration = (load_end_time - load_start_time) * 1e9  # Calculate load duration
    prompt_eval_count = len(query.split())  # Approximate token count for prompt
    prompt_eval_duration = (prompt_eval_end_time - prompt_eval_start_time) * 1e9  # Calculate prompt eval duration
    eval_count = len(result['result'].split())  # Approximate token count for response
    eval_duration = total_duration - load_duration - prompt_eval_duration  # Calculate actual eval duration
    
    # Calculate tokens per second
    tokens_per_second = (eval_count / eval_duration) * 1e9 if eval_duration > 0 else 0
    
    # Log to CSV
    filename = f"response_metrics_{datetime.now().strftime('%Y%m%d')}.csv"
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # If file is empty, write header
            writer.writerow(["timestamp", "query", "total_duration", "load_duration", "prompt_eval_count", 
                             "prompt_eval_duration", "eval_count", "eval_duration", "tokens_per_second"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query,
            total_duration,
            load_duration,
            prompt_eval_count,
            prompt_eval_duration,
            eval_count,
            eval_duration,
            tokens_per_second
        ])

# Custom CSS to style the app
st.markdown("""
<style>
    .stApp {
        max-width: 100%;
        padding: 1rem;
    }
    .main .block-container {
        max-width: 100%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
    }
    h1 {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.2rem;
        padding: 0.5rem 1rem;
    }
    .stButton > button {
        font-size: 1.2rem;
        padding: 0.5rem 2rem;
        margin-top: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e6f3ff;
        align-items: flex-end;
    }
    .chat-message.bot {
        background-color: #f0f0f0;
        align-items: flex-start;
    }
    .chat-message .message-content {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Streamlit UI for chatbot
st.title("CounselorAI")

# Sidebar for model selection
model = st.sidebar.selectbox("Your model", ["llama3-Q4-latest"])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Query input
query = st.chat_input("You:")
if query:
    # Display user message in chat message container
    st.chat_message("user").markdown(query)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.spinner("Processing query..."):
        start_time = time.time()
        
        load_start_time = time.time()
        qa_chain = get_qa_chain()  # This includes loading the model and vector store
        load_end_time = time.time()
        
        prompt_eval_start_time = time.time()
        result = qa_chain({"query": query})
        prompt_eval_end_time = time.time()
        
        end_time = time.time()
        
        # Start a background thread to calculate and log metrics
        threading.Thread(target=calculate_and_log_metrics, args=(query, result, start_time, end_time, 
                                                                 load_start_time, load_end_time, 
                                                                 prompt_eval_start_time, prompt_eval_end_time)).start()
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(result['result'])
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": result['result']})
        
        with st.expander("View Source Documents"):
            for i, doc in enumerate(result['source_documents'], 1):
                st.markdown(f"**Source {i}:**")
                st.write(f"Content: {doc.page_content[:300]}...")
                st.write(f"Source: {doc.metadata.get('source', 'Unknown')}")
                st.markdown("---")

# Footer
st.markdown("---")
