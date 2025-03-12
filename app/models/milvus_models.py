from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union


class TextWithMetadata(BaseModel):
    """Text and associated metadata."""
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchTextWithMetadata(BaseModel):
    """Batch of texts with metadata for insertion."""
    items: List[TextWithMetadata]
    collection_name: Optional[str] = None  # Optional target collection


class MilvusInsertResponse(BaseModel):
    """Response for insertion operation."""
    ids: List[Union[str, int]]
    insert_count: int = 0


class MilvusSearchRequest(BaseModel):
    """Request for vector search."""
    texts: List[str]
    collection_name: Optional[str] = None
    limit: int = 5
    output_fields: Optional[List[str]] = None
    filter: Optional[str] = None


class MilvusRangeSearchRequest(BaseModel):
    """Request for vector search with range parameters."""
    texts: List[str]
    collection_name: Optional[str] = None
    limit: int = 5
    radius: float = 0.4
    range_filter: float = 0.9
    output_fields: Optional[List[str]] = None
    filter: Optional[str] = None


class MilvusDeleteRequest(BaseModel):
    """Request for deletion operation."""
    collection_name: Optional[str] = None
    ids: Optional[List[Union[str, int]]] = None
    filter: Optional[str] = None


class MilvusQueryRequest(BaseModel):
    """Request for query operation."""
    collection_name: Optional[str] = None
    ids: Optional[List[Union[str, int]]] = None
    filter: Optional[str] = None
    output_fields: Optional[List[str]] = None


class CreateCollectionRequest(BaseModel):
    """Request for collection creation."""
    collection_name: str
    dimension: int = Field(default=256)
    metric_type: str = Field(default="COSINE")


class CollectionResponse(BaseModel):
    """Response for collection operations."""
    status: str
    message: str


class CollectionsListResponse(BaseModel):
    """Response for listing collections."""
    collections: List[str]


class CollectionCountResponse(BaseModel):
    """Response for count operation."""
    collection: Optional[str] = None
    count: int