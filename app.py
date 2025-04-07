from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai

app = Flask(__name__)
load_dotenv()

# Load keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load model and connect to Pinecone
embedder = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("medicalbot")

# Load Gemini
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")

def retrieve_context(query, k=5):
    vector = embedder.encode(query).tolist()
    response = index.query(vector=vector, top_k=k, include_metadata=True)
    context = "\n\n".join([match["metadata"]["text"] for match in response["matches"]])
    return context

def generate_answer(query, context):
    prompt = f"""
You are a medical assistant. Use the following context to answer the user's question.

Context:
{context}

Question: {query}

Answer:"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_query = request.form["query"]
        context = retrieve_context(user_query)
        response = generate_answer(user_query, context)
        return render_template("index.html", response=response, query=user_query)
    return render_template("index.html", response=None)

# Run the app
if __name__ == "__main__":
    port = 5000
    app.run(host="0.0.0.0", port=port)
