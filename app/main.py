from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
from typing import AsyncGenerator

from app.populate_database import populate_database
from app.query_data import query_rag_stream
from app.config import DEFAULT_MODEL, AVAILABLE_MODELS

app = FastAPI()

# Serve static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# Pydantic schema for incoming POST body
class QueryRequest(BaseModel):
    query_text: str
    model: str  # User will always select one

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": AVAILABLE_MODELS,
        "default_model": DEFAULT_MODEL
    })

@app.post("/populate_database")
async def populate_db():
    try:
        async def event_generator():
            async for msg in populate_database():
                yield msg.encode("utf-8")

        return StreamingResponse(event_generator(), media_type="text/plain")
    except Exception as e:
        print(f"Error in populate_database endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(request: QueryRequest):
    try:
        # Model is taken from user selection in frontend
        generator = query_rag_stream(request.query_text, request.model)

        def event_stream():
            for chunk in generator:
                yield chunk

        return StreamingResponse(event_stream(), media_type="text/plain")
    except Exception as e:
        print(f"Error in query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
