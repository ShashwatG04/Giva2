from fastapi import FastAPI, Query
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv
import os
import requests

load_dotenv(find_dotenv())

NEWSAPI_TOKEN = os.environ.get("NEWS_API_KEY")
NEWS_API_URL = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWSAPI_TOKEN}"


app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

dimension = 384 # Embedding dimension for all_Mini-LM
index = faiss.IndexFlatL2(dimension)

try:
    index = faiss.read_index("index.faiss")
    documents = np.load("article_list.npy", allow_pickle=True).tolist()
except:
    documents = []
    
class QueryRequest(BaseModel):
    query: str
    top_k: int  = 5
    metric: str = "cosine"
    
@app.get("/")
def home():
    return {"message":"Document Similarity API running on Hugging Face Spaces."}

@app.post("/api/search")
def search_docs(request: QueryRequest):    
    query_embedding = model.encode([request.query]).astype("float32")
    
    if request.metric == "cosine":
        faiss.normalize_L2(query_embedding) # Normalizing the embedding if the user metric is cosine
    elif request.metric == "dot":
        query_embedding = query_embedding
    else:
        return {"error": "Wrong metric. Default metric: cosine"}
    
        
    D,I = index.search(query_embedding, request.top_k)
    
    results = [{"document": documents[i]} for i in I[0]]
    return {"query": request.query, "results": results}
    
    
@app.post("/api/add_document")
def add_document():
    global documents
    
    response = requests.get(NEWS_API_URL)
    
    if response.status_code != 200:
        return {"error": "Failed to fetch news articles"}
    
    news_data = response.json()
    articles = news_data.get("articles",[])
    
    if not articles:
        return {"message" : "No new articles found"}
    
    added_count = 0
    
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        content = article.get("content", "")
        
        text = f"{title}. {description}. {content}"
        
        if text in documents:
            continue
        
        new_embedding  = model.encode([text]).astype("float32")
        faiss.normalize_L2(new_embedding)
        
        index.add(new_embedding)
        
        documents.append(text)
        added_count += 1
        
    faiss.write_index(index, "index.faiss")
    np.save("article_list.npy", np.array(documents, dtype = object))
    
    return {"message": f"{added_count} new articles added!", "total_documents": len(documents)}


