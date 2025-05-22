import google.generativeai as genai
from app.config import GOOGLE_API_KEY, GEMINI_EMBED_MODEL

genai.configure(api_key=GOOGLE_API_KEY)

class GeminiEmbeddings:
    def __init__(self, model_name=GEMINI_EMBED_MODEL):
        self.model_name = model_name

    def embed_query(self, text: str) -> list[float]:
        response = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query"
        )
        return response["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            response = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(response["embedding"])
        return embeddings

def get_embedding_function():
    return GeminiEmbeddings()
