from fastapi import APIRouter, Depends, HTTPException
import logging

from utils.milvus_manager import MilvusManager
from services.embedding_service import EmbeddingService
from services.milvus_service import MilvusService
from models.milvus_models import (
    CreateCollectionRequest, 
    CollectionResponse,
    CollectionsListResponse,
    CollectionCountResponse
)
from core.dependencies import get_milvus_manager, get_tokenizer, get_ort_session

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Milvus Collections"], prefix="/milvus/collections")


@router.get("", response_model=CollectionsListResponse)
async def list_collections(
    milvus_manager: MilvusManager = Depends(get_milvus_manager)
):
    """List all collections in the Milvus database."""
    try:
        # Get all collections
        collections = milvus_manager.list_collections()
        
        return {"collections": collections}
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")


@router.post("", response_model=CollectionResponse)
async def create_collection(
    request: CreateCollectionRequest,
    milvus_manager: MilvusManager = Depends(get_milvus_manager)
):
    """Create a new collection in the Milvus database."""
    try:
        # Create the collection with metric_type using the correct approach
        success = milvus_manager.create_collection(
            collection_name=request.collection_name,
            dimension=request.dimension,
            metric_type=request.metric_type
        )
        
        if success:
            return {"status": "success", "message": f"Collection {request.collection_name} created successfully with metric_type={request.metric_type}"}
        else:
            return {"status": "warning", "message": f"Collection {request.collection_name} already exists"}
    except Exception as e:
        logger.error(f"Error creating collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating collection: {str(e)}")


@router.delete("/{collection_name}", response_model=CollectionResponse)
async def drop_collection(
    collection_name: str,
    milvus_manager: MilvusManager = Depends(get_milvus_manager)
):
    """Drop a collection from the Milvus database."""
    try:
        # Drop the collection
        success = milvus_manager.drop_collection(collection_name=collection_name)
        
        if success:
            return {"status": "success", "message": f"Collection {collection_name} dropped successfully"}
        else:
            return {"status": "warning", "message": f"Collection {collection_name} does not exist"}
    except Exception as e:
        logger.error(f"Error dropping collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error dropping collection: {str(e)}")


@router.get("/{collection_name}")
async def describe_collection(
    collection_name: str,
    milvus_manager: MilvusManager = Depends(get_milvus_manager)
):
    """Get information about a collection."""
    try:
        # Get collection info
        info = milvus_manager.describe_collection(collection_name=collection_name)
        
        if info is None:
            raise HTTPException(status_code=404, detail=f"Collection {collection_name} not found")
            
        return {"collection_info": info}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error describing collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error describing collection: {str(e)}")


@router.get("/{collection_name}/count", response_model=CollectionCountResponse)
async def count_collection(
    collection_name: str,
    milvus_manager: MilvusManager = Depends(get_milvus_manager)
):
    """Get the number of embeddings in a specific Milvus collection."""
    try:
        # Get count from specified collection
        count = milvus_manager.count(collection_name=collection_name)
        
        return {"collection": collection_name, "count": count}
    except Exception as e:
        logger.error(f"Error getting count from collection {collection_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting count from collection {collection_name}: {str(e)}")