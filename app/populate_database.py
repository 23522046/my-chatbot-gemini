import os
import shutil
import time

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from app.get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = "data"

# Main entry for FastAPI
async def populate_database():
    clear_database()  # Step 1: Clean existing data
    documents = load_documents()
    chunks = split_documents(documents)
    result = add_to_chroma(chunks)
    return result

# Step 2: Load PDFs from /data
def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()

# Step 3: Split into chunks
def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)

# Step 4: Add to Chroma (skip existing IDs)
def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)
    print(f"✅ {len(chunks_with_ids)} chunks created")

    existing = db.get(include=[])
    existing_ids = set(existing["ids"])

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        new_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_ids)
        print(f"🆕 {len(new_chunks)} new chunks added to ChromaDB")
        return len(new_chunks)
    else:
        print("✅ No new documents to add")
        return 0

# Unique ID generator for each chunk
def calculate_chunk_ids(chunks: list[Document]):
    last_page_id = None
    chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            chunk_index += 1
        else:
            chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{chunk_index}"
        last_page_id = current_page_id

    return chunks

# Utility: Wipe ChromaDB folder
def clear_database():
    if os.path.exists(CHROMA_PATH):
        tmp_path = CHROMA_PATH + "_to_delete"
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path, ignore_errors=True)
        os.rename(CHROMA_PATH, tmp_path)
        time.sleep(1)  # Give OS a moment
        shutil.rmtree(tmp_path)
        print("🧹 Cleared existing ChromaDB")
