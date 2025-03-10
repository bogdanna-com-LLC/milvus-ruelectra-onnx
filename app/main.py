from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from model import get_embedding_model

app = FastAPI(title="ruElectra Embeddings API")


class TextInput(BaseModel):
    texts: List[str]


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int


@app.on_event("startup")
async def startup_event():
    # Load model at startup to avoid loading it for each request
    get_embedding_model()


@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(input_data: TextInput):
    """
    Generate embeddings for the provided texts
    """
    try:
        model = get_embedding_model()
        embeddings = model.generate_embeddings(input_data.texts)

        # Convert to Python list for JSON serialization
        embeddings_list = embeddings.tolist()

        return {
            "embeddings": embeddings_list,
            "dimensions": embeddings.shape[1]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "healthy"}