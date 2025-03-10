from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Dict, Any

from .gpu_utils import GPUManager

# Initialize FastAPI application
app = FastAPI(
    title="GPU Information Service",
    description="A simple FastAPI service to retrieve GPU information and perform computations"
)

# Initialize GPU Manager
gpu_manager = GPUManager()


class ComputationRequest(BaseModel):
    """
    Pydantic model for computation request
    """
    matrix_size: int = 10000


@app.get("/gpu/info")
async def get_gpu_information() -> Dict[str, Any]:
    """
    Endpoint to retrieve GPU system information

    :return: Dictionary with GPU details
    """
    return gpu_manager.get_gpu_info()


@app.post("/gpu/compute")
async def perform_gpu_computation(
    request: ComputationRequest = ComputationRequest()
) -> Dict[str, Any]:
    """
    Perform a compute-intensive task on GPU

    :param request: Computation request with matrix size
    :return: Computation metrics
    """
    return gpu_manager.perform_gpu_computation(request.matrix_size)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint

    :return: Service status
    """
    return {
        "status": "healthy",
        "message": "GPU Information Service is running"
    }


# Optional: Run directly for testing
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)