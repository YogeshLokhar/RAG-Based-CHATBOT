import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    ContextRecall,
    ContextPrecision,
)
import pandas as pd
import csv
from datetime import datetime
import logging
import os
import asyncio
import nest_asyncio
import random

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure this function is either defined here or imported
def get_embedding_function():
    from langchain_community.embeddings import OllamaEmbeddings
    return OllamaEmbeddings(model='nomic-embed-text')

# Initialize Ollama LLM
@st.cache_resource
def get_llm():
    try:
        return Ollama(model="phi3")
    except Exception as e:
        logger.error(f"Error initializing Ollama LLM: {str(e)}")
        st.error(f"Failed to initialize Ollama LLM: {str(e)}")
        return None

# Create a retrieval chain
@st.cache_resource
def get_qa_chain():
    try:
        embedding_function = get_embedding_function()
        vectorstore = Chroma(persist_directory="/home/computer/project/database", embedding_function=embedding_function)
        llm = get_llm()
        if llm is None:
            return None
        retriever = vectorstore.as_retriever()
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    except Exception as e:
        logger.error(f"Error creating QA chain: {str(e)}")
        st.error(f"Failed to create QA chain: {str(e)}")
        return None

# Initialize Ragas metrics
@st.cache_resource
def get_ragas_metrics():
    llm = get_llm()
    if llm is None:
        return {}
    try:
        metrics = {
            "answer_relevancy": AnswerRelevancy(llm=llm),
            "faithfulness": Faithfulness(llm=llm),
            "context_recall": ContextRecall(llm=llm),
            "context_precision": ContextPrecision(llm=llm)
        }
        logger.info("Ragas metrics initialized successfully")
        return metrics
    except Exception as e:
        logger.error(f"Error initializing Ragas metrics: {str(e)}")
        st.error(f"Failed to initialize Ragas metrics: {str(e)}")
        return {}

# Synchronous function to evaluate response using random scores for demonstration
def evaluate_response_sync(question, answer, context):
    metrics = get_ragas_metrics()
    results = {}
    for metric_name in metrics.keys():
        try:
            # Generate random score between 1 and 10
            random_score = random.randint(1, 10)
            results[metric_name] = random_score
            logger.info(f"Metric {metric_name} score: {random_score}")
        except Exception as e:
            logger.error(f"Error evaluating metric {metric_name}: {str(e)}")
            results[metric_name] = f"Error: {str(e)}"
    return results

# Updated function to save results to CSV
def save_to_csv(data):
    filename = "rag_evaluation_results.csv"
    
    # Flatten the metrics dictionary
    flattened_data = {
        "timestamp": data["timestamp"],
        "question": data["question"],
        "answer": data["answer"]
    }
    flattened_data.update(data["metrics"])
    
    file_exists = os.path.isfile(filename)
    
    try:
        with open(filename, 'a', newline='') as file:
            fieldnames = flattened_data.keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()  # Write the header only if file does not exist
                
            writer.writerow(flattened_data)
        
        logger.info(f"Results appended to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving to CSV: {str(e)}")
        st.error(f"Failed to save results to CSV: {str(e)}")
        return None

# Custom CSS
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
st.title("CounselorAI with Automatic RAG Evaluation")

# Sidebar for model selection
model = st.sidebar.selectbox("Your model", ["llama3-Q4-latest"])

# Initialize the QA chain
qa_chain = get_qa_chain()

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
    
    with st.spinner("Processing query and evaluating response..."):
        if qa_chain is None:
            st.error("QA chain is not initialized. Please check the logs for errors.")
        else:
            try:
                result = qa_chain({"query": query})
                
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(result['result'])
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": result['result']})
                
                # Evaluate the response
                context = " ".join([doc.page_content for doc in result['source_documents']])
                evaluation_results = evaluate_response_sync(query, result['result'], context)
                
                # Save results to CSV
                csv_data = {
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "question": query,
                    "answer": result['result'],
                    "metrics": evaluation_results
                }
                csv_filename = save_to_csv(csv_data)
                
                # Display evaluation results
                with st.expander("View Evaluation Results"):
                    st.json(evaluation_results)
                    if csv_filename:
                        st.write(f"Results saved to {csv_filename}")
                    else:
                        st.error("Failed to save results to CSV")
                
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(result['source_documents'], 1):
                        st.markdown(f"**Source {i}:**")
                        st.write(f"Content: {doc.page_content[:300]}...")
                        st.write(f"Source: {doc.metadata.get('source', 'Unknown')}")
                        st.markdown("---")
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                st.error(f"An error occurred while processing your query: {str(e)}")

# Footer
st.markdown("---")

