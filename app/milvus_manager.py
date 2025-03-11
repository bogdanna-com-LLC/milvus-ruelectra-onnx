import os
import logging
from typing import List, Dict, Any, Optional, Union
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MilvusManager:
    def __init__(
        self, 
        db_path: str = "milvus_data.db", 
        collection_name: str = "embeddings_collection",
        dimension: int = 256,
        metric_type: str = "COSINE",
        recreate: bool = False
    ):
        """
        Initialize the Milvus Manager with the given parameters.
        
        Args:
            db_path: Path to the Milvus database file
            collection_name: Name of the collection to use
            dimension: Dimension of the embeddings vectors
            metric_type: Distance metric type for vector similarity (COSINE, L2, IP, etc.)
            recreate: Whether to recreate the collection if it exists
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.dimension = dimension
        self.metric_type = metric_type
        self.client = None
        
        try:
            # Initialize the Milvus client
            logger.info(f"Initializing Milvus client with database at {db_path}")
            self.client = MilvusClient(db_path)
            
            # Check if collection exists and handle accordingly
            if self.client.has_collection(collection_name=self.collection_name):
                if recreate:
                    logger.info(f"Dropping existing collection {collection_name}")
                    self.client.drop_collection(collection_name=self.collection_name)
                    self._create_collection()
                else:
                    logger.info(f"Using existing collection {collection_name}")
                    # Get collection info to verify configuration
                    try:
                        collection_info = self.client.describe_collection(collection_name=self.collection_name)
                        logger.info(f"Collection info: {collection_info}")
                    except Exception as e:
                        logger.warning(f"Failed to get collection info: {str(e)}")
            else:
                logger.info(f"Collection {collection_name} does not exist, creating it")
                self._create_collection()
                
        except Exception as e:
            logger.error(f"Failed to initialize Milvus: {str(e)}")
            raise RuntimeError(f"Failed to initialize Milvus: {str(e)}")
    
    def _create_collection(self):
        """Create a new collection with the specified parameters."""
        try:
            logger.info(f"Creating collection {self.collection_name} with dimension {self.dimension}")
            
            # Define fields with explicit schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
            ]
            
            # Create schema
            schema = CollectionSchema(fields)
            
            # Create collection with specified schema and metric type
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                metric_type=self.metric_type
            )
            
            logger.info(f"Collection {self.collection_name} created successfully")
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise RuntimeError(f"Failed to create collection: {str(e)}")
    
    def create_collection(self, collection_name: str, dimension: int, metric_type: str = "COSINE"):
        """
        Create a new collection with the given parameters.
        
        Args:
            collection_name: Name for the new collection
            dimension: Dimension of the embeddings vectors
            metric_type: Distance metric type for vector similarity
            
        Returns:
            True if successful, raises an exception otherwise
        """
        if not self.client:
            raise RuntimeError("Milvus client not initialized")
            
        try:
            if self.client.has_collection(collection_name=collection_name):
                logger.warning(f"Collection {collection_name} already exists")
                return False
            
            # Define fields with explicit schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
            ]
            
            # Create schema
            schema = CollectionSchema(fields)
            
            # Create collection with specified schema and metric type
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                metric_type=metric_type
            )
            
            logger.info(f"Collection {collection_name} created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {str(e)}")
            raise RuntimeError(f"Failed to create collection {collection_name}: {str(e)}")
    
    def list_collections(self):
        """
        List all collections in the database.
        
        Returns:
            List of collection names
        """
        if not self.client:
            raise RuntimeError("Milvus client not initialized")
            
        try:
            collections = self.client.list_collections()
            logger.info(f"Found {len(collections)} collections")
            return collections
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}")
            raise RuntimeError(f"Failed to list collections: {str(e)}")
    
    def drop_collection(self, collection_name: str):
        """
        Drop a collection from the database.
        
        Args:
            collection_name: Name of the collection to drop
            
        Returns:
            True if successful, raises an exception otherwise
        """
        if not self.client:
            raise RuntimeError("Milvus client not initialized")
            
        try:
            if not self.client.has_collection(collection_name=collection_name):
                logger.warning(f"Collection {collection_name} does not exist")
                return False
                
            self.client.drop_collection(collection_name=collection_name)
            logger.info(f"Collection {collection_name} dropped successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to drop collection {collection_name}: {str(e)}")
            raise RuntimeError(f"Failed to drop collection {collection_name}: {str(e)}")
    
    def describe_collection(self, collection_name: str):
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection information
        """
        if not self.client:
            raise RuntimeError("Milvus client not initialized")
            
        try:
            if not self.client.has_collection(collection_name=collection_name):
                logger.warning(f"Collection {collection_name} does not exist")
                return None
                
            info = self.client.describe_collection(collection_name=collection_name)
            logger.info(f"Retrieved information for collection {collection_name}")
            return info
        except Exception as e:
            logger.error(f"Failed to describe collection {collection_name}: {str(e)}")
            raise RuntimeError(f"Failed to describe collection {collection_name}: {str(e)}")
    
    def insert_embeddings(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]], collection_name: Optional[str] = None) -> List[str]:
        """
        Insert embeddings into the collection with associated metadata.
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries (must have same length as embeddings)
            collection_name: Optional name of the collection to insert into (defaults to current collection)
            
        Returns:
            List of inserted IDs
        """
        if not self.client:
            raise RuntimeError("Milvus client not initialized")
            
        col_name = collection_name or self.collection_name
            
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have the same length")
            
        if not embeddings:
            logger.warning("No embeddings to insert")
            return []
        
        # Check if collection exists
        if not self.client.has_collection(collection_name=col_name):
            logger.error(f"Collection {col_name} does not exist")
            raise ValueError(f"Collection {col_name} does not exist")
            
        # Get collection info to check dimension
        try:
            collection_info = self.client.describe_collection(collection_name=col_name)
            # In a real implementation, we would verify the dimension here
            logger.info(f"Inserting into collection: {col_name}")
        except Exception as e:
            logger.warning(f"Could not verify collection dimensions: {str(e)}")
            
        # Validate embedding dimensions (using self.dimension as fallback)
        for i, embedding in enumerate(embeddings):
            if len(embedding) != self.dimension:
                raise ValueError(f"Embedding at index {i} has dimension {len(embedding)}, expected {self.dimension}")
        
        try:
            # Prepare data for insertion - explicitly DO NOT include 'id' field
            data = []
            for i in range(len(embeddings)):
                # Start with the vector
                entry = {"vector": embeddings[i]}
                
                # Add metadata fields (excluding 'id' if present)
                for key, value in metadata[i].items():
                    if key != 'id':  # Skip 'id' field as it will be auto-generated
                        entry[key] = value
                
                data.append(entry)
            
            # Insert the data
            logger.info(f"Inserting {len(data)} embeddings into collection {col_name}")
            result = self.client.insert(
                collection_name=col_name,
                data=data
            )
            
            # Log the insert result
            if isinstance(result, dict):
                insert_count = result.get('insert_count', 0)
                ids = result.get('ids', [])
                logger.info(f"Inserted {insert_count} embeddings successfully")
                return ids
            else:
                logger.warning(f"Unexpected insert result format: {result}")
                return result if isinstance(result, list) else []
            
        except Exception as e:
            logger.error(f"Failed to insert embeddings: {str(e)}")
            raise RuntimeError(f"Failed to insert embeddings: {str(e)}")
    
    def search(
        self, 
        query_embeddings: List[List[float]], 
        limit: int = 5, 
        output_fields: Optional[List[str]] = None,
        filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.
        
        Args:
            query_embeddings: List of query embedding vectors
            limit: Maximum number of results to return per query
            output_fields: Fields to include in the results
            filter: Optional filter expression for metadata filtering
            
        Returns:
            List of search results
        """
        if not self.client:
            raise RuntimeError("Milvus client not initialized")
            
        if not query_embeddings:
            logger.warning("No query embeddings provided")
            return []
            
        # Validate embedding dimensions
        for i, embedding in enumerate(query_embeddings):
            if len(embedding) != self.dimension:
                raise ValueError(f"Query embedding at index {i} has dimension {len(embedding)}, expected {self.dimension}")
        
        try:
            logger.info(f"Searching collection {self.collection_name} with {len(query_embeddings)} queries")
            search_results = self.client.search(
                collection_name=self.collection_name,
                data=query_embeddings,
                limit=limit,
                output_fields=output_fields,
                filter=filter
            )
            
            logger.info(f"Search completed with {len(search_results)} result sets")
            return search_results
        except Exception as e:
            logger.error(f"Failed to search embeddings: {str(e)}")
            raise RuntimeError(f"Failed to search embeddings: {str(e)}")
    
    def delete(self, ids: Optional[List[Union[str, int]]] = None, filter: Optional[str] = None) -> List[Union[str, int]]:
        """
        Delete embeddings by their IDs or using a filter expression.
        
        Args:
            ids: List of embedding IDs to delete
            filter: Filter expression to match entities to delete
            
        Returns:
            List of deleted IDs
        """
        if not self.client:
            raise RuntimeError("Milvus client not initialized")
            
        if not ids and not filter:
            logger.warning("No IDs or filter provided for deletion")
            return []
            
        try:
            if ids:
                logger.info(f"Deleting {len(ids)} embeddings from collection {self.collection_name}")
                result = self.client.delete(
                    collection_name=self.collection_name,
                    ids=ids
                )
            else:
                logger.info(f"Deleting embeddings with filter '{filter}' from collection {self.collection_name}")
                result = self.client.delete(
                    collection_name=self.collection_name,
                    filter=filter
                )
            
            logger.info(f"Deletion completed successfully: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {str(e)}")
            raise RuntimeError(f"Failed to delete embeddings: {str(e)}")
    
    def query(
        self, 
        ids: Optional[List[Union[str, int]]] = None, 
        filter: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query entities by IDs or filter expression.
        
        Args:
            ids: List of entity IDs to query
            filter: Filter expression to match entities
            output_fields: Fields to include in the results
            
        Returns:
            List of query results
        """
        if not self.client:
            raise RuntimeError("Milvus client not initialized")
            
        if not ids and not filter:
            logger.warning("No IDs or filter provided for query")
            return []
        
        try:
            if ids:
                logger.info(f"Querying {len(ids)} entities from collection {self.collection_name}")
                result = self.client.query(
                    collection_name=self.collection_name,
                    ids=ids,
                    output_fields=output_fields
                )
            else:
                logger.info(f"Querying entities with filter '{filter}' from collection {self.collection_name}")
                result = self.client.query(
                    collection_name=self.collection_name,
                    filter=filter,
                    output_fields=output_fields
                )
            
            logger.info(f"Query completed with {len(result)} results")
            return result
        except Exception as e:
            logger.error(f"Failed to query entities: {str(e)}")
            raise RuntimeError(f"Failed to query entities: {str(e)}")
    
    def count(self, collection_name: Optional[str] = None) -> int:
        """
        Get the number of entities in the collection.
        
        Args:
            collection_name: Name of the collection to count (defaults to current collection)
            
        Returns:
            Number of entities
        """
        if not self.client:
            raise RuntimeError("Milvus client not initialized")
        
        col_name = collection_name or self.collection_name
            
        try:
            count = self.client.count(collection_name=col_name)
            logger.info(f"Collection {col_name} has {count} entities")
            return count
        except Exception as e:
            logger.error(f"Failed to get count: {str(e)}")
            raise RuntimeError(f"Failed to get count: {str(e)}")
    
    def close(self):
        """Close the Milvus client connection."""
        # Currently, MilvusClient doesn't have an explicit close method
        # The database connection is managed by the MilvusClient internally
        logger.info("Milvus manager resources released")
        self.client = None