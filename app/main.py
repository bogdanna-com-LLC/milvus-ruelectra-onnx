# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import os

app = FastAPI(title="ruElectra Embeddings API with ONNX Runtime")

# Load tokenizer and ONNX session
model_path = os.environ.get("MODEL_PATH", "/app/models/ruelectra_small.onnx")
tokenizer_path = os.environ.get("TOKENIZER_PATH", "/app/models/onnx_tokenizer")


# Define models for API
class TextInput(BaseModel):
    texts: List[str]


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int


# Initialize tokenizer and model globally
tokenizer = None
ort_session = None


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
    """Initialize tokenizer and ONNX runtime session at startup."""
    global tokenizer, ort_session

    # Check if CUDA is available and set provider accordingly
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(model_path, providers=providers)

        # Log available providers
        print(f"Available providers: {ort.get_available_providers()}")
        print(f"Using providers: {ort_session.get_providers()}")

    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        raise RuntimeError(f"Failed to initialize model: {str(e)}")


@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(input_data: TextInput):
    """Generate embeddings for the provided texts."""
    global tokenizer, ort_session

    if tokenizer is None or ort_session is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        # Tokenize the input texts
        encoded_input = tokenizer(
            input_data.texts,
            padding=True,
            truncation=True,
            max_length=24,
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

        return {
            "embeddings": sentence_embeddings.tolist(),
            "dimensions": sentence_embeddings.shape[1]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": "loaded" if ort_session is not None else "not loaded"}