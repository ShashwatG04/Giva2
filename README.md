# Giva2
Document Similarity Search API
(Shashwat Gautam)


This is the second task of the Giva assignment. The project builds an API that fetches news articles, creates document embeddings, and performs similarity search.

Steps to Run the Code Locally

->Download the Repository:
Download all files in the repository and save them locally.

->Get a NewsAPI Key:

Visit newsapi.org and generate an API key.

For evaluation purposes, a key is provided in the .env file.
Note: You can update .env with your own API key if needed.

->Install Dependencies:
Install the required dependencies by running:
pip install fastapi pydantic faiss-cpu numpy sentence-transformers python-dotenv requests

Alternatively, if you have a requirements.txt file, run:
pip install -r requirements.txt

->To start the API Server run the following command in your terminal:

uvicorn main:app --host 0.0.0.0 --port 8000

This will start the app on localhost at port 8000.

->API Endpoints
GET /
Returns a simple message indicating that the API is running.

->POST /api/add_document
Fetches the latest news from NewsAPI and adds the documents (with embeddings) to the database.

->POST /api/search
Accepts a JSON payload with the following keys:

query: The search text.

top_k: (Optional) Number of similar documents to return (default is 5).

metric: (Optional) Similarity metric, either "cosine" or "dot" (default is "cosine").

Testing the API
I have tested this API using Thunderclient in VSCode. You can use any API testing tool (e.g., Postman) or a Python client (see search.py) to test the endpoints.
