from fastapi import APIRouter, Depends, HTTPException
from transformers import AutoTokenizer
import onnxruntime as ort
import logging

from models.embedding_models import TextInput, EmbeddingResponse
from services.embedding_service import EmbeddingService
from core.dependencies import get_tokenizer, get_ort_session

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Embeddings"], prefix="/embeddings")


@router.post("", response_model=EmbeddingResponse)
async def create_embeddings(
    input_data: TextInput,
    tokenizer: AutoTokenizer = Depends(get_tokenizer),
    ort_session: ort.InferenceSession = Depends(get_ort_session)
):
    """Generate embeddings for the provided texts."""
    try:
        # Create service
        embedding_service = EmbeddingService(tokenizer, ort_session)
        
        # Generate embeddings
        sentence_embeddings = embedding_service.generate_embeddings(input_data.texts)

        return {
            "embeddings": sentence_embeddings.tolist(),
            "dimensions": sentence_embeddings.shape[1]
        }
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")