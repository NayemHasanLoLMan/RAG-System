import os
import pickle
import fitz  # PyMuPDF for PDF extraction
from dotenv import load_dotenv
from openai import OpenAI
import faiss
import numpy as np

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FAISS index file paths
FAISS_INDEX_PATH = "harry_potter_faiss.index"
METADATA_PATH = "harry_potter_metadata.pkl"

def create_or_load_faiss_index(dimension=1536):
    """Create a new FAISS index or load existing one."""
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing FAISS index...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
        print(f"Loaded index with {index.ntotal} vectors")
    else:
        print("Creating new FAISS index...")
        index = faiss.IndexFlatL2(dimension)  # L2 distance
        metadata = []
    
    return index, metadata

def save_faiss_index(index, metadata):
    """Save FAISS index and metadata to disk."""
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved FAISS index with {index.ntotal} vectors")

def extract_pdf_text(pdf_path):
    """Extract text from PDF page by page."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    pdf_text = []
    
    print(f"Extracting text from {doc.page_count} pages...")
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text.strip():
            pdf_text.append(text)
        else:
            pdf_text.append('')
    
    doc.close()
    
    title = doc.metadata.get("title", os.path.basename(pdf_path))
    return pdf_text, title

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into chunks with overlap."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk)
        
        start += chunk_size - overlap
    
    return chunks

def embed_text_with_openai(text):
    """Generate embeddings using OpenAI's text-embedding-3-small model."""
    if not text.strip():
        return None
    
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def upload_pdf_to_faiss(pdf_path, chunk_size=1000, overlap=200):
    """Upload PDF to FAISS index chunk by chunk."""
    # Load or create FAISS index
    index, metadata = create_or_load_faiss_index()
    
    # Extract text from PDF
    pdf_text, pdf_title = extract_pdf_text(pdf_path)
    
    print(f"\nProcessing PDF: {pdf_title}")
    print(f"Total pages: {len(pdf_text)}")
    
    total_chunks = 0
    successful_uploads = 0
    
    # Process each page
    for page_num, page_text in enumerate(pdf_text):
        if not page_text.strip():
            print(f"Skipping empty page {page_num + 1}")
            continue
        
        # Split page into chunks
        chunks = chunk_text(page_text, chunk_size, overlap)
        
        print(f"Page {page_num + 1}: {len(chunks)} chunks")
        
        # Process each chunk
        for chunk_idx, chunk in enumerate(chunks):
            total_chunks += 1
            print(f"  Processing chunk {chunk_idx + 1}/{len(chunks)}...", end="\r")
            
            # Generate embedding
            embedding = embed_text_with_openai(chunk)
            
            if embedding is not None:
                # Convert to numpy array
                embedding_array = np.array([embedding], dtype='float32')
                
                # Add to FAISS index
                index.add(embedding_array)
                
                # Store metadata
                chunk_metadata = {
                    "pdf_title": pdf_title,
                    "page_number": page_num + 1,
                    "chunk_index": chunk_idx,
                    "text": chunk,
                    "char_count": len(chunk)
                }
                metadata.append(chunk_metadata)
                
                successful_uploads += 1
            else:
                print(f"\n  Failed to embed chunk {chunk_idx + 1}")
        
        print(f"  Page {page_num + 1} complete: {len(chunks)} chunks uploaded")
    
    # Save the index and metadata
    save_faiss_index(index, metadata)
    
    print(f"\n{'='*60}")
    print(f"Upload Summary:")
    print(f"  Successfully uploaded: {successful_uploads} chunks")
    print(f"  Total chunks processed: {total_chunks}")
    print(f"  Total vectors in index: {index.ntotal}")
    print(f"{'='*60}")

def search_faiss(query, top_k=5):
    """Search FAISS index for similar chunks."""
    if not os.path.exists(FAISS_INDEX_PATH):
        print("No FAISS index found. Please upload a PDF first.")
        return []
    
    # Load index and metadata
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    
    # Generate query embedding
    query_embedding = embed_text_with_openai(query)
    if query_embedding is None:
        return []
    
    # Convert to numpy array
    query_array = np.array([query_embedding], dtype='float32')
    
    # Search
    distances, indices = index.search(query_array, top_k)
    
    # Retrieve results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            result = metadata[idx].copy()
            result['distance'] = float(distances[0][i])
            results.append(result)
    
    return results

def main():
    """Main function to run the PDF upload process."""
    pdf_path = r"C:\Users\hasan\Downloads\New project\HP1 - Harry Potter and the Sorcerer_s Stone.pdf"
    
    try:
        upload_pdf_to_faiss(pdf_path, chunk_size=1000, overlap=200)
        print("\n✓ PDF successfully uploaded to FAISS index!")
        
        # Example search
        print("\n" + "="*60)
        print("Testing search functionality...")
        results = search_faiss("What is Hogwarts?", top_k=3)
        
        if results:
            print("\nTop 3 search results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Page {result['page_number']}, Chunk {result['chunk_index']}")
                print(f"   Distance: {result['distance']:.4f}")
                print(f"   Text preview: {result['text'][:200]}...")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise

if __name__ == "__main__":
    main()