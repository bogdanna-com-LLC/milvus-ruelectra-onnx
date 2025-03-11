import logging
from typing import List, Dict, Any, Optional, Union

from utils.milvus_manager import MilvusManager
from services.embedding_service import EmbeddingService
from models.milvus_models import TextWithMetadata, BatchTextWithMetadata

logger = logging.getLogger(__name__)


class MilvusService:
    """Service for Milvus operations."""
    
    def __init__(self, milvus_manager: MilvusManager, embedding_service: EmbeddingService):
        """
        Initialize the Milvus service.
        
        Args:
            milvus_manager: Milvus manager for database operations
            embedding_service: Embedding service for text encoding
        """
        self.milvus_manager = milvus_manager
        self.embedding_service = embedding_service
    
    # Collection management
    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        return self.milvus_manager.list_collections()
    
    def create_collection(self, collection_name: str, dimension: int, metric_type: str = "COSINE") -> bool:
        """Create a new collection."""
        return self.milvus_manager.create_collection(collection_name, dimension, metric_type)
    
    def drop_collection(self, collection_name: str) -> bool:
        """Drop a collection."""
        return self.milvus_manager.drop_collection(collection_name)
    
    def describe_collection(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get collection information."""
        return self.milvus_manager.describe_collection(collection_name)
    
    def count_collection(self, collection_name: Optional[str] = None) -> int:
        """Count entities in a collection."""
        return self.milvus_manager.count(collection_name)
    
    # Data operations
    def insert_data(self, batch_data: BatchTextWithMetadata) -> List[Union[str, int]]:
        """
        Generate embeddings and insert into Milvus.
        
        Args:
            batch_data: Batch of texts with metadata
            
        Returns:
            List of inserted IDs
        """
        # Extract texts and metadata
        texts = [item.text for item in batch_data.items]
        metadata_list = [item.metadata for item in batch_data.items]
        
        # Generate embeddings
        embeddings = self.embedding_service.generate_embeddings(texts)
        
        # Update metadata with text field if not present
        for i, metadata in enumerate(metadata_list):
            if "text" not in metadata:
                metadata["text"] = texts[i]
        
        # Insert into Milvus
        return self.milvus_manager.insert_embeddings(
            embeddings=embeddings.tolist(),
            metadata=metadata_list,
            collection_name=batch_data.collection_name
        )
    
    def search(
        self, 
        texts: List[str], 
        collection_name: Optional[str] = None,
        limit: int = 5, 
        output_fields: Optional[List[str]] = None,
        filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.
        
        Args:
            texts: List of query texts
            collection_name: Optional target collection
            limit: Maximum number of results per query
            output_fields: Fields to include in results
            filter: Optional filter expression
            
        Returns:
            List of search results
        """
        # Generate embeddings for query texts
        query_embeddings = self.embedding_service.generate_embeddings(texts)
        
        # Search in Milvus
        return self.milvus_manager.search(
            query_embeddings=query_embeddings.tolist(),
            collection_name=collection_name,
            limit=limit,
            output_fields=output_fields,
            filter=filter
        )
    
    def delete(
        self, 
        collection_name: Optional[str] = None,
        ids: Optional[List[Union[str, int]]] = None, 
        filter: Optional[str] = None
    ) -> List[Union[str, int]]:
        """
        Delete entities from the collection.
        
        Args:
            collection_name: Optional target collection
            ids: Optional list of entity IDs to delete
            filter: Optional filter expression
            
        Returns:
            List of deleted IDs
        """
        return self.milvus_manager.delete(
            collection_name=collection_name,
            ids=ids,
            filter=filter
        )
    
    def query(
        self, 
        collection_name: Optional[str] = None,
        ids: Optional[List[Union[str, int]]] = None, 
        filter: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query entities from the collection.
        
        Args:
            collection_name: Optional target collection
            ids: Optional list of entity IDs to query
            filter: Optional filter expression
            output_fields: Fields to include in results
            
        Returns:
            List of query results
        """
        return self.milvus_manager.query(
            collection_name=collection_name,
            ids=ids,
            filter=filter,
            output_fields=output_fields
        )