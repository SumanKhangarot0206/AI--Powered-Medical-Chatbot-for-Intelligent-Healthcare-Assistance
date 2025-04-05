import streamlit as st
import os
from pinecone import Pinecone
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables only once
@st.cache_resource
def load_env():
    load_dotenv()
    return os.getenv("PINECONE_API_KEY"), os.getenv("GOOGLE_API_KEY")

# Load SentenceTransformer model only once
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Pinecone only once
@st.cache_resource
def connect_to_pinecone(api_key):
    pc = Pinecone(api_key=api_key)
    return pc.Index("medicalbot")

# Set up Gemini model once
@st.cache_resource
def load_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-pro-latest")


# Get top-k documents from Pinecone
def retrieve_context(query, embedder, index, k=5):
    vector = embedder.encode(query).tolist()
    response = index.query(vector=vector, top_k=k, include_metadata=True)
    context = "\n\n".join([match["metadata"]["text"] for match in response["matches"]])
    return context

# Generate response from Gemini
def generate_answer(model, query, context):
    prompt = f"""
You are a medical assistant. Use the following context to answer the user's question.

Context:
{context}

Question: {query}

Answer:"""
    response = model.generate_content(prompt)
    return response.text.strip()

# --- MAIN APP ---
st.set_page_config(page_title="Medical Bot", page_icon="🩺")
st.title("🤖 AI Medical Assistant")

pinecone_api_key, gemini_api_key = load_env()
embedder = load_embedder()
index = connect_to_pinecone(pinecone_api_key)
gemini_model = load_gemini(gemini_api_key)

user_query = st.text_input("Ask your medical question:")

if user_query:
    with st.spinner("Retrieving information and generating response..."):
        context = retrieve_context(user_query, embedder, index)
        response = generate_answer(gemini_model, user_query, context)
    st.subheader("🧠 AI Response:")
    st.write(response)