from fastapi import APIRouter, Depends, HTTPException
import logging
from typing import Optional
from transformers import AutoTokenizer
import onnxruntime as ort

from utils.milvus_manager import MilvusManager
from services.embedding_service import EmbeddingService
from services.milvus_service import MilvusService
from models.milvus_models import (
    BatchTextWithMetadata,
    MilvusInsertResponse,
    MilvusSearchRequest,
    MilvusRangeSearchRequest,
    MilvusDeleteRequest,
    MilvusQueryRequest,
    CollectionCountResponse
)
from core.dependencies import get_milvus_manager, get_tokenizer, get_ort_session

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Milvus Data Operations"], prefix="/milvus")


@router.post("/insert", response_model=MilvusInsertResponse)
async def insert_to_milvus(
    input_data: BatchTextWithMetadata,
    milvus_manager: MilvusManager = Depends(get_milvus_manager),
    tokenizer: AutoTokenizer = Depends(get_tokenizer),
    ort_session: ort.InferenceSession = Depends(get_ort_session)
):
    """Generate embeddings and insert into Milvus."""
    try:
        # Create services
        embedding_service = EmbeddingService(tokenizer, ort_session)
        milvus_service = MilvusService(milvus_manager, embedding_service)
        
        # Insert data
        ids = milvus_service.insert_data(input_data)
        
        return {"ids": ids, "insert_count": len(ids)}
    except Exception as e:
        logger.error(f"Error inserting embeddings to Milvus: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error inserting embeddings to Milvus: {str(e)}")


@router.post("/search")
async def search_milvus(
    search_request: MilvusSearchRequest,
    milvus_manager: MilvusManager = Depends(get_milvus_manager),
    tokenizer: AutoTokenizer = Depends(get_tokenizer),
    ort_session: ort.InferenceSession = Depends(get_ort_session)
):
    """Search for similar embeddings in Milvus."""
    try:
        # Create services
        embedding_service = EmbeddingService(tokenizer, ort_session)
        milvus_service = MilvusService(milvus_manager, embedding_service)
        
        # Search in Milvus
        results = milvus_service.search(
            texts=search_request.texts,
            collection_name=search_request.collection_name,
            limit=search_request.limit,
            output_fields=search_request.output_fields,
            filter=search_request.filter
        )
        
        return results
    except Exception as e:
        logger.error(f"Error searching in Milvus: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching in Milvus: {str(e)}")


@router.post("/search/range")
async def search_milvus_with_range(
    search_request: MilvusRangeSearchRequest,
    milvus_manager: MilvusManager = Depends(get_milvus_manager),
    tokenizer: AutoTokenizer = Depends(get_tokenizer),
    ort_session: ort.InferenceSession = Depends(get_ort_session)
):
    """Search for similar embeddings in Milvus with range parameters."""
    try:
        # Create services
        embedding_service = EmbeddingService(tokenizer, ort_session)
        milvus_service = MilvusService(milvus_manager, embedding_service)
        
        # Search in Milvus with range parameters
        results = milvus_service.search_with_range(
            texts=search_request.texts,
            collection_name=search_request.collection_name,
            limit=search_request.limit,
            radius=search_request.radius,
            range_filter=search_request.range_filter,
            output_fields=search_request.output_fields,
            filter=search_request.filter
        )
        
        return results
    except Exception as e:
        logger.error(f"Error searching in Milvus with range parameters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching in Milvus with range parameters: {str(e)}")


@router.post("/delete")
async def delete_from_milvus(
    delete_request: MilvusDeleteRequest,
    milvus_manager: MilvusManager = Depends(get_milvus_manager)
):
    """Delete embeddings from Milvus by ID or filter."""
    try:
        # Delete from Milvus
        deleted_ids = milvus_manager.delete(
            collection_name=delete_request.collection_name,
            ids=delete_request.ids, 
            filter=delete_request.filter
        )
        
        return {
            "deleted": deleted_ids, 
            "message": f"Successfully deleted {len(deleted_ids)} entities"
        }
    except Exception as e:
        logger.error(f"Error deleting from Milvus: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting from Milvus: {str(e)}")


@router.post("/query")
async def query_milvus(
    query_request: MilvusQueryRequest,
    milvus_manager: MilvusManager = Depends(get_milvus_manager)
):
    """Query entities from Milvus by ID or filter."""
    try:
        # Query from Milvus
        results = milvus_manager.query(
            collection_name=query_request.collection_name,
            ids=query_request.ids,
            filter=query_request.filter,
            output_fields=query_request.output_fields
        )
        
        return results
    except Exception as e:
        logger.error(f"Error querying from Milvus: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying from Milvus: {str(e)}")


@router.get("/count", response_model=CollectionCountResponse)
async def count_milvus(
    collection_name: Optional[str] = None,
    milvus_manager: MilvusManager = Depends(get_milvus_manager)
):
    """Get the number of embeddings in the specified Milvus collection."""
    try:
        # Get count from Milvus
        count = milvus_manager.count(collection_name)
        
        return {"collection": collection_name, "count": count}
    except Exception as e:
        logger.error(f"Error getting count from Milvus: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting count from Milvus: {str(e)}")