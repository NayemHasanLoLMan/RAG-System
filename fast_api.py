import os
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import faiss
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FAISS paths
FAISS_INDEX_PATH = "harry_potter_faiss.index"
METADATA_PATH = "harry_potter_metadata.pkl"

# Load FAISS index and metadata
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, 'rb') as f:
    metadata = pickle.load(f)

# Initialize FastAPI
app = FastAPI()

# Request model
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

# Response model
class SearchResult(BaseModel):
    page_number: int
    chunk_index: int
    text: str
    distance: float

def embed_text(text):
    """Generate embedding using OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

@app.get("/")
def read_root():
    return {
        "message": "Harry Potter FAISS Search API",
        "total_chunks": index.ntotal
    }

@app.post("/search")
def search(request: QueryRequest):
    """Search for similar text chunks."""
    # Generate query embedding
    query_embedding = embed_text(request.query)
    query_array = np.array([query_embedding], dtype='float32')
    
    # Search FAISS index
    distances, indices = index.search(query_array, request.top_k)
    
    # Build results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            results.append({
                "page_number": metadata[idx]["page_number"],
                "chunk_index": metadata[idx]["chunk_index"],
                "text": metadata[idx]["text"],
                "distance": float(distances[0][i])
            })
    
    return {
        "query": request.query,
        "results": results
    }

@app.get("/stats")
def get_stats():
    """Get database statistics."""
    return {
        "total_vectors": index.ntotal,
        "total_metadata": len(metadata),
        "dimension": index.d
    }