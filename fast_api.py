import json
import faiss
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware

# Local imports
from utils import embed_texts, get_chat_response # Use the shared batch embedder
from config import FAISS_INDEX_PATH, METADATA_PATH


SYSTEM_PROMPT = """
You are an expert assistant on Harry Potter, using the text of "Harry Potter and the Sorcerer's Stone".
You will be given a user's question and several context chunks from the book.
Your task is to answer the user's question *using only the provided context*.

- If the context contains the answer, synthesize a clear and concise response from it.
- Do not make up information or use any external knowledge.
- If the context chunks do not contain enough information to answer the question, politely state that the answer is not in the provided document sections.
- Do not mention the context chunks directly in your response (e.g., "According to the context..."). Just answer the question.
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager to load resources on startup.
    This is the modern way to handle startup/shutdown events.
    """
    print("INFO:     Loading FAISS index and metadata...")
    try:
        app.state.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            app.state.metadata = json.load(f)
        
        print("INFO:     Successfully loaded index and metadata.")
        print(f"INFO:     Total vectors in index: {app.state.faiss_index.ntotal}")
        print(f"INFO:     Total metadata entries: {len(app.state.metadata)}")
        
    except FileNotFoundError:
        print("WARNING:  FAISS index or metadata file not found.")
        print("WARNING:  Please run the ingestion script (new.py) first.")
        app.state.faiss_index = None
        app.state.metadata = None
    except Exception as e:
        print(f"ERROR:    Error loading resources: {e}")
        app.state.faiss_index = None
        app.state.metadata = None
    
    yield
    
    # Clean up resources if needed (e.g., close db connections)
    print("INFO:     Shutting down...")

# Initialize FastAPI app with the lifespan manager
app = FastAPI(lifespan=lifespan)

# Enable CORS for local development so browser preflight (OPTIONS) requests succeed.
# In production, narrow allow_origins to your actual frontend origin(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request and response
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class SearchResult(BaseModel):
    page_number: int
    chunk_index: int
    text: str
    distance: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: List[ChatMessage] = []
    top_k: int = 3

class ChatResponse(BaseModel):
    answer: str
    history: List[ChatMessage]

def read_root(request: Request):
    """Helper function to get loaded resources from app state."""
    if not request.app.state.faiss_index or not request.app.state.metadata:
        raise HTTPException(
            status_code=503, 
            detail="FAISS index is not loaded. Please run the ingestion script."
        )
    index = request.app.state.faiss_index
    return {
        "message": "Harry Potter FAISS Search API",
        "total_chunks": index.ntotal
    }

@app.post("/search", response_model=SearchResponse)
def search(request: Request, query: SearchRequest):
    """Search for similar text chunks (Vector Search Only)."""
    if not request.app.state.faiss_index or not request.app.state.metadata:
        raise HTTPException(
            status_code=503, 
            detail="FAISS index is not loaded. Please run the ingestion script."
        )
    index = request.app.state.faiss_index
    metadata = request.app.state.metadata
    
    query_embedding_list = embed_texts([query.query])
    
    if not query_embedding_list:
        raise HTTPException(status_code=500, detail="Failed to embed query text.")
    
    query_array = np.array(query_embedding_list, dtype='float32')
    
    distances, indices = index.search(query_array, query.top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(metadata):
            meta = metadata[idx]
            results.append(SearchResult(
                page_number=meta["page_number"],
                chunk_index=meta["chunk_index"],
                text=meta["text"],
                distance=float(distances[0][i])
            ))
    
    return SearchResponse(query=query.query, results=results)


@app.post("/chat", response_model=ChatResponse)
def chat(request: Request, chat_request: ChatRequest):
    """
    Chat with the document, using RAG and conversation history.
    """
    if not request.app.state.faiss_index or not request.app.state.metadata:
        raise HTTPException(
            status_code=503, 
            detail="FAISS index is not loaded. Please run the ingestion script."
        )
    index = request.app.state.faiss_index
    metadata = request.app.state.metadata
    
    # 1. RAG - RETRIEVAL
    # Embed the user's *current* query to find relevant context
    query_embedding_list = embed_texts([chat_request.query])
    if not query_embedding_list:
        raise HTTPException(status_code=500, detail="Failed to embed query text.")
    
    query_array = np.array(query_embedding_list, dtype='float32')
    
    # Search FAISS index
    distances, indices = index.search(query_array, chat_request.top_k)
    
    # Get the text chunks from metadata
    context_chunks = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(metadata):
            context_chunks.append(metadata[idx]["text"])
    
    context_str = "\n\n---\n\n".join(context_chunks)
    
    # 2. RAG - GENERATION
    
    # Build the list of messages for the AI
    # We start with the system prompt
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add the conversation history
    for msg in chat_request.history:
        messages.append(msg.dict())
        
    # Add the user's new query, *prefixed with the retrieved context*
    formatted_query = f"""
Here is the context from the book:
---
{context_str}
---

Now, answer this question: {chat_request.query}
"""
    messages.append({"role": "user", "content": formatted_query})

    # Get the response from the chat model
    ai_answer = get_chat_response(messages)
    
    if not ai_answer:
        raise HTTPException(status_code=500, detail="Failed to get a response from the AI model.")
    
    # 3. UPDATE HISTORY
    # We add the *original* user query (not the context-stuffed one)
    # and the AI's answer to the history
    new_history = chat_request.history + [
        ChatMessage(role="user", content=chat_request.query),
        ChatMessage(role="assistant", content=ai_answer)
    ]

    return ChatResponse(answer=ai_answer, history=new_history)


@app.get("/chat")
def chat_info():
    """GET helper for browser users.

    Returns usage instructions. The actual chat API expects POST /chat with a JSON
    body: {"query":"...","history":[],"top_k":3}. Browsers will send an
    OPTIONS preflight for cross-origin POSTs — CORS is enabled above to allow this
    during local testing.
    """
    return {
        "message": "This endpoint expects POST. Use POST /chat with JSON body:\n{\"query\": \"your question\", \"history\": [], \"top_k\": 3}",
        "note": "CORS is enabled for local testing. In production, restrict origins."
    }

@app.get("/stats")
def get_stats(request: Request):
    """Get database statistics."""
    if not request.app.state.faiss_index or not request.app.state.metadata:
        raise HTTPException(
            status_code=503, 
            detail="FAISS index is not loaded. Please run the ingestion script."
        )
    index = request.app.state.faiss_index
    metadata = request.app.state.metadata
    return {
        "total_vectors": index.ntotal,
        "total_metadata": len(metadata),
        "dimension": index.d
    }