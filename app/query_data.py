import re
import google.generativeai as genai
from app.get_embedding_function import get_embedding_function
from app.config import GOOGLE_API_KEY, GEMINI_EMBED_MODEL

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Utility to remove <think>...</think> block from response
def remove_think_block(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# Main function to stream response from Gemini with RAG context
def query_rag_stream(query_text: str, model_name: str):
    # Configure Gemini
    genai.configure(api_key=GOOGLE_API_KEY)

    # Load Chroma DB and embedding function
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Find relevant context from vector store
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Build prompt with context
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("🔎 Prompt sent to Gemini:\n", prompt)

    # Send prompt to Gemini model
    model = genai.GenerativeModel(model_name)
    stream = model.generate_content(prompt, stream=True)

    # Stream filtering: remove <think> while yielding content
    raw_response = ""
    user_response = ""
    in_think_block = False

    for chunk in stream:
        part = chunk.text or ""
        raw_response += part

        if "<think>" in part:
            in_think_block = True
            continue
        if "</think>" in part:
            in_think_block = False
            continue
        if in_think_block:
            continue

        user_response += part
        yield part

    print("🧠 Raw Gemini response:\n", raw_response)
    print("✅ Final response sent to client:\n", user_response)
