# main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import os
import logging
from milvus_manager import MilvusManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ruElectra Embeddings API with ONNX Runtime (CPU) and Milvus")

# Configuration
model_path = os.environ.get("MODEL_PATH", "/app/models/ruelectra_small.onnx")
tokenizer_path = os.environ.get("TOKENIZER_PATH", "/app/models/onnx_tokenizer")
milvus_db_path = os.environ.get("MILVUS_DB_PATH", "/app/data/milvus_data.db")
milvus_collection = os.environ.get("MILVUS_COLLECTION", "embeddings_collection")
embedding_dimension = int(os.environ.get("EMBEDDING_DIMENSION", "256"))
max_token_length = int(os.environ.get("MAX_TOKEN_LENGTH", "512"))
milvus_metric_type = os.environ.get("MILVUS_METRIC_TYPE", "COSINE")

logger.info(f"Model path: {model_path}")
logger.info(f"Tokenizer path: {tokenizer_path}")
logger.info(f"Milvus DB path: {milvus_db_path}")
logger.info(f"Milvus collection: {milvus_collection}")
logger.info(f"Embedding dimension: {embedding_dimension}")
logger.info(f"Max token length: {max_token_length}")
logger.info(f"Milvus metric type: {milvus_metric_type}")

# Define models for API
class TextInput(BaseModel):
    texts: List[str]


class TextWithMetadata(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchTextWithMetadata(BaseModel):
    items: List[TextWithMetadata]
    collection_name: Optional[str] = None  # Optional target collection


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int


class MilvusInsertResponse(BaseModel):
    ids: List[Union[str, int]]
    insert_count: int = 0


class MilvusSearchRequest(BaseModel):
    texts: List[str]
    limit: int = 5
    output_fields: Optional[List[str]] = None
    filter: Optional[str] = None


class MilvusDeleteRequest(BaseModel):
    ids: Optional[List[Union[str, int]]] = None
    filter: Optional[str] = None


class MilvusQueryRequest(BaseModel):
    ids: Optional[List[Union[str, int]]] = None
    filter: Optional[str] = None
    output_fields: Optional[List[str]] = None


class CreateCollectionRequest(BaseModel):
    collection_name: str
    dimension: int = Field(default=256)
    metric_type: str = Field(default="COSINE")


# Initialize tokenizer, model, and milvus globally
tokenizer = None
ort_session = None
milvus_manager = None


def mean_pooling(token_embeddings, attention_mask):
    """Perform mean pooling on token embeddings with attention mask."""
    # Convert attention mask to float and create expanded mask
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
    # Sum embeddings with attention mask applied
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    # Sum mask values (avoiding division by zero)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    # Calculate mean
    return sum_embeddings / sum_mask


@app.on_event("startup")
async def startup_event():
    """Initialize tokenizer, ONNX runtime session, and Milvus at startup."""
    global tokenizer, ort_session, milvus_manager

    # Use CPU execution provider only
    providers = ['CPUExecutionProvider']
    
    # Log available providers
    logger.info(f"Available ONNX providers: {ort.get_available_providers()}")
    logger.info(f"Using providers: {providers}")

    try:
        # Initialize tokenizer and model
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        if not os.path.exists(tokenizer_path):
            logger.error(f"Tokenizer directory not found: {tokenizer_path}")
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_path}")
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info("Tokenizer loaded successfully")

        logger.info("Loading ONNX model...")
        ort_session = ort.InferenceSession(model_path, providers=providers)
        logger.info("ONNX model loaded successfully")
        
        # Initialize Milvus
        logger.info("Initializing Milvus...")
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(milvus_db_path), exist_ok=True)
        
        milvus_manager = MilvusManager(
            db_path=milvus_db_path,
            collection_name=milvus_collection,
            dimension=embedding_dimension,
            metric_type=milvus_metric_type,
            recreate=False  # Don't recreate the collection if it exists
        )
        logger.info("Milvus initialized successfully")
        
        # Log collection count
        try:
            count = milvus_manager.count()
            logger.info(f"Collection contains {count} embeddings")
        except Exception as e:
            logger.warning(f"Failed to get count: {str(e)}")

    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise RuntimeError(f"Failed to initialize: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global milvus_manager
    
    try:
        if milvus_manager:
            logger.info("Closing Milvus connection...")
            milvus_manager.close()
            logger.info("Milvus connection closed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


def generate_embeddings(texts):
    """Generate embeddings for the provided texts."""
    if tokenizer is None or ort_session is None:
        raise RuntimeError("Model not initialized")
        
    if not texts:
        return []
        
    # Tokenize the input texts
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_token_length,
        return_tensors='np'
    )

    # Get input names from the model
    input_names = [input.name for input in ort_session.get_inputs()]

    # Prepare inputs as dictionary, matching the expected names
    ort_inputs = {name: encoded_input[name] for name in input_names if name in encoded_input}

    # Run inference
    ort_outputs = ort_session.run(None, ort_inputs)

    # Get the hidden states (first output)
    token_embeddings = ort_outputs[0]

    # Apply mean pooling to get sentence embeddings
    sentence_embeddings = mean_pooling(token_embeddings, encoded_input['attention_mask'])
    
    return sentence_embeddings


@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(input_data: TextInput):
    """Generate embeddings for the provided texts."""
    global tokenizer, ort_session

    if tokenizer is None or ort_session is None:
        logger.error("Model not initialized")
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        # Generate embeddings
        sentence_embeddings = generate_embeddings(input_data.texts)

        return {
            "embeddings": sentence_embeddings.tolist(),
            "dimensions": sentence_embeddings.shape[1]
        }
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")


# Collection management endpoints
@app.get("/milvus/collections")
async def list_collections():
    """List all collections in the Milvus database."""
    global milvus_manager
    
    if milvus_manager is None:
        logger.error("Milvus not initialized")
        raise HTTPException(status_code=500, detail="Milvus not initialized")
        
    try:
        # Get all collections
        collections = milvus_manager.list_collections()
        
        return {"collections": collections}
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")


@app.post("/milvus/collections")
async def create_collection(request: CreateCollectionRequest):
    """Create a new collection in the Milvus database."""
    global milvus_manager
    
    if milvus_manager is None:
        logger.error("Milvus not initialized")
        raise HTTPException(status_code=500, detail="Milvus not initialized")
        
    try:
        # Create the collection
        success = milvus_manager.create_collection(
            collection_name=request.collection_name,
            dimension=request.dimension,
            metric_type=request.metric_type
        )
        
        if success:
            return {"status": "success", "message": f"Collection {request.collection_name} created successfully"}
        else:
            return {"status": "warning", "message": f"Collection {request.collection_name} already exists"}
    except Exception as e:
        logger.error(f"Error creating collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating collection: {str(e)}")


@app.delete("/milvus/collections/{collection_name}")
async def drop_collection(collection_name: str):
    """Drop a collection from the Milvus database."""
    global milvus_manager
    
    if milvus_manager is None:
        logger.error("Milvus not initialized")
        raise HTTPException(status_code=500, detail="Milvus not initialized")
        
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


@app.get("/milvus/collections/{collection_name}")
async def describe_collection(collection_name: str):
    """Get information about a collection."""
    global milvus_manager
    
    if milvus_manager is None:
        logger.error("Milvus not initialized")
        raise HTTPException(status_code=500, detail="Milvus not initialized")
        
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


@app.post("/milvus/insert", response_model=MilvusInsertResponse)
async def insert_to_milvus(input_data: BatchTextWithMetadata):
    """Generate embeddings and insert into Milvus."""
    global milvus_manager
    
    if milvus_manager is None:
        logger.error("Milvus not initialized")
        raise HTTPException(status_code=500, detail="Milvus not initialized")
        
    if not input_data.items:
        return {"ids": [], "insert_count": 0}
        
    try:
        # Extract texts and metadata
        texts = [item.text for item in input_data.items]
        metadata_list = [item.metadata for item in input_data.items]
        
        # Generate embeddings
        embeddings = generate_embeddings(texts)
        
        # Insert into Milvus
        for i, metadata in enumerate(metadata_list):
            # Add text to metadata if not already present
            if "text" not in metadata:
                metadata["text"] = texts[i]
        
        # Use specified collection if provided
        target_collection = input_data.collection_name
        if target_collection:
            logger.info(f"Using specified collection: {target_collection}")
        
        ids = milvus_manager.insert_embeddings(
            embeddings=embeddings.tolist(),
            metadata=metadata_list,
            collection_name=target_collection
        )
        
        return {"ids": ids, "insert_count": len(ids)}
    except Exception as e:
        logger.error(f"Error inserting embeddings to Milvus: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error inserting embeddings to Milvus: {str(e)}")


@app.post("/milvus/search")
async def search_milvus(search_request: MilvusSearchRequest):
    """Search for similar embeddings in Milvus."""
    global milvus_manager
    
    if milvus_manager is None:
        logger.error("Milvus not initialized")
        raise HTTPException(status_code=500, detail="Milvus not initialized")
        
    if not search_request.texts:
        return []
        
    try:
        # Generate embeddings for query texts
        query_embeddings = generate_embeddings(search_request.texts)
        
        # Search in Milvus
        results = milvus_manager.search(
            query_embeddings=query_embeddings.tolist(),
            limit=search_request.limit,
            output_fields=search_request.output_fields,
            filter=search_request.filter
        )
        
        return results
    except Exception as e:
        logger.error(f"Error searching in Milvus: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching in Milvus: {str(e)}")


@app.post("/milvus/delete")
async def delete_from_milvus(delete_request: MilvusDeleteRequest):
    """Delete embeddings from Milvus by ID or filter."""
    global milvus_manager
    
    if milvus_manager is None:
        logger.error("Milvus not initialized")
        raise HTTPException(status_code=500, detail="Milvus not initialized")
        
    if not delete_request.ids and not delete_request.filter:
        return {"deleted": [], "message": "No IDs or filter provided"}
        
    try:
        # Delete from Milvus
        deleted_ids = milvus_manager.delete(
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


@app.post("/milvus/query")
async def query_milvus(query_request: MilvusQueryRequest):
    """Query entities from Milvus by ID or filter."""
    global milvus_manager
    
    if milvus_manager is None:
        logger.error("Milvus not initialized")
        raise HTTPException(status_code=500, detail="Milvus not initialized")
        
    if not query_request.ids and not query_request.filter:
        return []
        
    try:
        # Query from Milvus
        results = milvus_manager.query(
            ids=query_request.ids,
            filter=query_request.filter,
            output_fields=query_request.output_fields
        )
        
        return results
    except Exception as e:
        logger.error(f"Error querying from Milvus: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying from Milvus: {str(e)}")


@app.get("/milvus/count")
async def count_milvus():
    """Get the number of embeddings in Milvus."""
    global milvus_manager
    
    if milvus_manager is None:
        logger.error("Milvus not initialized")
        raise HTTPException(status_code=500, detail="Milvus not initialized")
        
    try:
        # Get count from Milvus
        count = milvus_manager.count()
        
        return {"count": count}
    except Exception as e:
        logger.error(f"Error getting count from Milvus: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting count from Milvus: {str(e)}")


@app.get("/milvus/count/{collection_name}")
async def count_collection(collection_name: str):
    """Get the number of embeddings in a specific Milvus collection."""
    global milvus_manager
    
    if milvus_manager is None:
        logger.error("Milvus not initialized")
        raise HTTPException(status_code=500, detail="Milvus not initialized")
        
    try:
        # Get count from specified collection
        count = milvus_manager.count(collection_name=collection_name)
        
        return {"collection": collection_name, "count": count}
    except Exception as e:
        logger.error(f"Error getting count from collection {collection_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting count from collection {collection_name}: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_status = "loaded" if ort_session is not None else "not loaded"
    milvus_status = "connected" if milvus_manager is not None else "not connected"
    
    logger.info(f"Health check: Model {model_status}, Milvus {milvus_status}")
    
    return {
        "status": "healthy", 
        "model": model_status,
        "milvus": milvus_status
    }