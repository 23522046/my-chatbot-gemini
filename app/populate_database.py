import os
import shutil
import asyncio
from typing import AsyncGenerator

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from app.get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def clear_database():
    if os.path.exists(CHROMA_PATH):
        tmp_path = CHROMA_PATH + "_to_delete"
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path, ignore_errors=True)
        os.rename(CHROMA_PATH, tmp_path)
        # Kasih delay biar OS sempat handle rename
        import time; time.sleep(1)
        shutil.rmtree(tmp_path)
        print("🧹 Cleared existing ChromaDB")


def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()


def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)


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


def add_to_chroma_generator(chunks: list[Document], batch_size=20):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)
    total_chunks = len(chunks_with_ids)

    existing = db.get(include=[])
    existing_ids = set(existing["ids"])

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    total_new = len(new_chunks)

    if total_new == 0:
        yield "✅ No new documents to add\n"
        return

    yield f"Total new chunks to add: {total_new}\n"

    for i in range(0, total_new, batch_size):
        batch = new_chunks[i:i+batch_size]
        batch_ids = [chunk.metadata["id"] for chunk in batch]
        db.add_documents(batch, ids=batch_ids)
        yield f"🆕 Added chunks {i+1}-{min(i+batch_size, total_new)} of {total_new}\n"


async def populate_database() -> AsyncGenerator[str, None]:
    yield "Starting database population...\n"
    await asyncio.sleep(0)

    yield "Clearing existing ChromaDB...\n"
    await asyncio.to_thread(clear_database)
    yield "ChromaDB cleared.\n"

    yield "Loading documents from data directory...\n"
    documents = await asyncio.to_thread(load_documents)
    yield f"Documents loaded: {len(documents)}\n"

    yield "Splitting documents into chunks...\n"
    chunks = await asyncio.to_thread(split_documents, documents)
    yield f"Documents split into {len(chunks)} chunks\n"

    yield "Adding chunks to ChromaDB...\n"

    def sync_generator():
        return add_to_chroma_generator(chunks)

    # Jalankan di thread dan yield per update
    for msg in await asyncio.to_thread(lambda: list(sync_generator())):
        yield msg

    yield "Database population complete.\n"
