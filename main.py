from fastapi import FastAPI, Query
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv
import os
import requests

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set up News API details
NEWSAPI_TOKEN = os.environ.get("NEWS_API_KEY")
NEWS_API_URL = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWSAPI_TOKEN}"

# Initialize FastAPI app
app = FastAPI()

# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set embedding dimension for all-MiniLM-L6-v2
dimension = 384
# Initialize FAISS index
index = faiss.IndexFlatL2(dimension)

# Try to load existing index and documents; otherwise, initialize empty list
try:
    index = faiss.read_index("index.faiss")
    documents = np.load("article_list.npy", allow_pickle=True).tolist()
except Exception:
    documents = []

# Define the request schema for search endpoint
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    metric: str = "cosine"

@app.get("/")
def home():
    """Home endpoint to check API status."""
    return {"message": "Document Similarity API running on Hugging Face Spaces."}

@app.post("/api/search")
def search_docs(request: QueryRequest):
    # Generate embedding for the query
    query_embedding = model.encode([request.query]).astype("float32")
    
    # Normalize embedding if using cosine similarity
    if request.metric == "cosine":
        faiss.normalize_L2(query_embedding)
    elif request.metric == "dot":
        pass  # No changes required for dot product
    else:
        return {"error": "Wrong metric. Default metric: cosine"}
    
    # Perform similarity search in the index
    distances, indices = index.search(query_embedding, request.top_k)
    
    # Retrieve documents corresponding to the indices
    results = [{"document": documents[i]} for i in indices[0]]
    return {"query": request.query, "results": results}

@app.post("/api/add_document")
def add_document():
    global documents
    
    # Fetch news articles from NewsAPI
    response = requests.get(NEWS_API_URL)
    
    if response.status_code != 200:
        return {"error": "Failed to fetch news articles"}
    
    news_data = response.json()
    articles = news_data.get("articles", [])
    
    if not articles:
        return {"message": "No new articles found"}
    
    added_count = 0
    
    # Process each article
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        content = article.get("content", "")
        
        # Combine the article parts into one text
        text = f"{title}. {description}. {content}"
        
        # Skip if article already exists
        if text in documents:
            continue
        
        # Encode text into embedding and normalize
        new_embedding = model.encode([text]).astype("float32")
        faiss.normalize_L2(new_embedding)
        
        # Add new embedding to the index
        index.add(new_embedding)
        
        # Append document text to the list
        documents.append(text)
        added_count += 1
        
    # Save the updated index and documents list
    faiss.write_index(index, "index.faiss")
    np.save("article_list.npy", np.array(documents, dtype=object))
    
    return {"message": f"{added_count} new articles added!", "total_documents": len(documents)}
