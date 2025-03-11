from pydantic import BaseModel
from typing import List


class TextInput(BaseModel):
    """Input model for embedding generation."""
    texts: List[str]


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    embeddings: List[List[float]]
    dimensions: int
