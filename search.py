import requests

# URL of your FastAPI search endpoint
url = "http://localhost:8000/api/search"

# Take user input for the search query
user_query = input("Enter your search query: ")

# Prepare the payload with the user query
payload = {
    "query": user_query,
    "top_k": 5,
    "metric": "cosine"
}

# Set the request header to specify JSON content
headers = {"Content-Type": "application/json"}

# Send the POST request
response = requests.post(url, json=payload, headers=headers)

# Check the response and print results
if response.status_code == 200:
    print("Search Results:")
    print(response.json())
else:
    print("Error:", response.status_code, response.text)
