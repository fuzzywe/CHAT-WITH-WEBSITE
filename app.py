import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Load SentenceTransformer model (Hugging Face)
def get_embeddings_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Define embedding function
def embed_documents(texts):
    model = get_embeddings_model()
    return model.encode(texts).tolist()  # Ensure embeddings are a list of lists

# Function to get vectorstore from URL using Hugging Face's embeddings
def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # Create Chroma vector store with embedding function
    vector_store = Chroma(embedding_function=embed_documents)
    document_texts = [doc.page_content for doc in document_chunks]

    # Add texts with embeddings
    vector_store.add_texts(
        texts=document_texts,
        embeddings=embed_documents(document_texts)  # Provide embeddings here
    )
    
    return vector_store

# Retrieve context-aware information
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

# Conversation Retrieval-Augmented Generation (RAG) chain
def get_conversation_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}\nRestrict the answer to 10 words."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Get response from the system
def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation = get_conversation_rag_chain(retriever_chain)
    
    response = conversation.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']

# Streamlit app configuration
st.set_page_config(page_title="Chat with Websites", page_icon="🤖")
st.title("Chat with Websites")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")
else:
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am an AI bot. How can I help you?")
        ]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    # Input chat
    user_query = st.chat_input("Type your message here...")
    if user_query and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
