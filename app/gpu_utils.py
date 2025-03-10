import torch
import subprocess
from typing import Dict, Any, List


class GPUManager:
    """
    Utility class for managing and inspecting GPU resources
    """

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """
        Retrieve detailed GPU information using PyTorch and nvidia-smi

        :return: Dictionary with GPU details
        """
        gpu_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "devices": []
        }

        # Get PyTorch GPU details
        for i in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(i)
            gpu_info["devices"].append({
                "name": device.name,
                "total_memory": f"{device.total_memory / (1024 ** 2):.2f} MB",
                "compute_capability": f"{device.major}.{device.minor}"
            })

        # Attempt to get additional details via nvidia-smi
        try:
            nvidia_smi_output = subprocess.check_output(
                ["nvidia-smi"],
                universal_newlines=True
            )
            gpu_info["nvidia_smi_output"] = nvidia_smi_output
        except (subprocess.CalledProcessError, FileNotFoundError):
            gpu_info["nvidia_smi_output"] = "Could not retrieve nvidia-smi output"

        return gpu_info

    @staticmethod
    def perform_gpu_computation(matrix_size: int = 10000) -> Dict[str, Any]:
        """
        Perform a compute-intensive task on GPU

        :param matrix_size: Size of matrix for computation
        :return: Computation metrics
        """
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            # Create large matrices
            matrix_a = torch.randn(matrix_size, matrix_size, device=device)
            matrix_b = torch.randn(matrix_size, matrix_size, device=device)

            # Perform matrix multiplication
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()
            result = torch.matmul(matrix_a, matrix_b)
            end_time.record()

            # Synchronize and calculate timing
            torch.cuda.synchronize()
            computation_time = start_time.elapsed_time(end_time)

            return {
                "computation_device": str(device),
                "matrix_size": matrix_size,
                "computation_time_ms": computation_time,
                "result_norm": float(torch.norm(result).cpu())
            }
        except Exception as e:
            return {
                "error": str(e),
                "computation_device": str(device)
            }