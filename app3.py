import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Page configuration
st.set_page_config(page_title="C++ RAG Chatbot", page_icon="😉")
st.title("😉  C++ RAG Chatbot")
st.write("Ask any question about C++ introduction")

# Step 2: Load environment variables
load_dotenv()

# Step 3: Cache document loading
@st.cache_resource
def load_vector_store():
    loader = TextLoader("C++_Introduction.txt", encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    final_documents = text_splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(model_name="all-miniLM-l6-v2")

    db = FAISS.from_documents(final_documents, embedding)
    return db

# Load vector database
db = load_vector_store()

# Step 4: Query input
query = st.text_input("Enter your question about C++")
if query:
    documents = db.similarity_search(query, k=3)
    st.subheader("📝  Retrieved documents")
    for i, doc in enumerate(documents):
        st.markdown(f"**Result {i+1}:**")
        st.write(doc.page_content)
