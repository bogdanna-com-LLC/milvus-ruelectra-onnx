import numpy as np
import logging
from transformers import AutoTokenizer
import onnxruntime as ort
from typing import List

from core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings from text."""
    
    def __init__(self, tokenizer: AutoTokenizer, ort_session: ort.InferenceSession):
        """
        Initialize the embedding service.
        
        Args:
            tokenizer: Tokenizer for text processing
            ort_session: ONNX Runtime session for inference
        """
        self.tokenizer = tokenizer
        self.ort_session = ort_session
        self.max_token_length = settings.MAX_TOKEN_LENGTH
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """
        Perform mean pooling on token embeddings with attention mask.
        
        Args:
            token_embeddings: Embeddings from the model
            attention_mask: Attention mask for tokens
            
        Returns:
            Mean-pooled embeddings
        """
        # Convert attention mask to float and create expanded mask
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        # Sum embeddings with attention mask applied
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        # Sum mask values (avoiding division by zero)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        # Calculate mean
        return sum_embeddings / sum_mask
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for the provided texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
            
        # Tokenize the input texts
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_token_length,
            return_tensors='np'
        )

        # Get input names from the model
        input_names = [input.name for input in self.ort_session.get_inputs()]

        # Prepare inputs as dictionary, matching the expected names
        ort_inputs = {name: encoded_input[name] for name in input_names if name in encoded_input}

        # Run inference
        ort_outputs = self.ort_session.run(None, ort_inputs)

        # Get the hidden states (first output)
        token_embeddings = ort_outputs[0]

        # Apply mean pooling to get sentence embeddings
        sentence_embeddings = self.mean_pooling(token_embeddings, encoded_input['attention_mask'])
        
        return sentence_embeddings