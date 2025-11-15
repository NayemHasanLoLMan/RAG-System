import os
import json  # Use JSON instead of pickle
import argparse
import fitz  # PyMuPDF
import faiss
import numpy as np
from typing import List, Dict, Any

# Local imports
from utils import embed_texts
from config import (
    FAISS_INDEX_PATH, 
    METADATA_PATH, 
    EMBEDDING_DIMENSION
)

def create_fresh_faiss_index(dimension: int):
    """
    Deletes old index/metadata files and creates a new FAISS index.
    This prevents re-indexing the same data.
    """
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
        print(f"Removed old index file: {FAISS_INDEX_PATH}")
        
    if os.path.exists(METADATA_PATH):
        os.remove(METADATA_PATH)
        print(f"Removed old metadata file: {METADATA_PATH}")

    print("Creating new FAISS index...")
    # Using IndexFlatL2 for basic L2 (Euclidean) distance
    index = faiss.IndexFlatL2(dimension)
    metadata = []
    return index, metadata

def save_faiss_index(index: faiss.Index, metadata: List[Dict[str, Any]]):
    """Save FAISS index and metadata to disk (using JSON)."""
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved FAISS index with {index.ntotal} vectors.")
    print(f"Saved metadata for {len(metadata)} chunks.")

def extract_pdf_text(pdf_path: str):
    """Extract text from PDF, returning a list of (page_num, text)."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    pdf_pages = []
    
    print(f"Extracting text from {doc.page_count} pages...")
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text.strip():
            pdf_pages.append((page_num + 1, text))
    
    doc.close()
    
    title = doc.metadata.get("title", os.path.basename(pdf_path))
    return pdf_pages, title

def chunk_text(text: str, chunk_size=1000, overlap=200):
    """Split text into chunks with overlap."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= len(text):
            break
            
        start += chunk_size - overlap
    
    # Filter out any potential empty strings from chunking
    return [c for c in chunks if c.strip()]

def process_pdf_to_faiss(pdf_path: str, chunk_size: int, overlap: int, batch_size: int):
    """
    Full processing pipeline:
    1. Extracts text from PDF
    2. Chunks text
    3. Embeds chunks in batches
    4. Saves to FAISS and JSON metadata
    """
    index, metadata = create_fresh_faiss_index(EMBEDDING_DIMENSION)
    pdf_pages, pdf_title = extract_pdf_text(pdf_path)
    
    all_chunks = []
    all_metadata = []

    print(f"\nProcessing PDF: {pdf_title}")
    
    # Step 1 & 2: Extract and Chunk all pages
    for page_num, page_text in pdf_pages:
        if not page_text.strip():
            print(f"Skipping empty page {page_num}")
            continue
        
        chunks = chunk_text(page_text, chunk_size, overlap)
        print(f"Page {page_num}: Generated {len(chunks)} chunks")
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_metadata = {
                "pdf_title": pdf_title,
                "page_number": page_num,
                "chunk_index": chunk_idx,
                "text": chunk
            }
            all_chunks.append(chunk)
            all_metadata.append(chunk_metadata)
    
    print(f"\nTotal chunks generated: {len(all_chunks)}")
    if not all_chunks:
        print("No text chunks were generated. Exiting.")
        return

    # Step 3: Embed in Batches
    print(f"Embedding chunks in batches of {batch_size}...")
    successful_uploads = 0
    for i in range(0, len(all_chunks), batch_size):
        batch_texts = all_chunks[i:i+batch_size]
        batch_metadata = all_metadata[i:i+batch_size]
        
        print(f"  Processing batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}...")
        
        embeddings = embed_texts(batch_texts)
        
        if embeddings and len(embeddings) == len(batch_texts):
            embedding_array = np.array(embeddings, dtype='float32')
            index.add(embedding_array)
            metadata.extend(batch_metadata) # Add metadata for the successful batch
            successful_uploads += len(batch_texts)
        else:
            print(f"  Failed to embed batch {i//batch_size + 1}. Skipping.")
    
    # Step 4: Save
    if successful_uploads > 0:
        save_faiss_index(index, metadata)
    else:
        print("No chunks were successfully embedded. Index not saved.")

    print(f"\n{'='*60}")
    print(f"Upload Summary:")
    print(f"  Successfully uploaded: {successful_uploads} chunks")
    print(f"  Total chunks generated: {len(all_chunks)}")
    print(f"  Total vectors in index: {index.ntotal}")
    print(f"{'='*60}")

def main():
    """Main function to run the PDF upload process."""
    # This fixes the hardcoded path issue by using command-line arguments
    parser = argparse.ArgumentParser(description="Upload a PDF to a FAISS index.")
    
    # This line defines the argument as "pdf_path"
    parser.add_argument("pdf_path", type=str, help="The file path to the PDF document.")
    
    parser.add_argument("--chunk_size", type=int, default=1000, help="Character size for each text chunk.")
    parser.add_argument("--overlap", type=int, default=200, help="Character overlap between chunks.")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of chunks to embed in one API call.")
    
    args = parser.parse_args()
    
    try:
        # This line now correctly accesses "args.pdf_path"
        process_pdf_to_faiss(args.pdf_path, args.chunk_size, args.overlap, args.batch_size)
        print("\n✓ PDF successfully processed and indexed!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
    except Exception as e:
        print(f"\n✗ An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main()