# convert_to_onnx.py
import torch
from transformers import AutoTokenizer, AutoModel
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_electra_to_onnx():
    try:
        logger.info("Checking for required packages...")
        # Verify ONNX is installed
        try:
            import onnx
            logger.info(f"ONNX version: {onnx.__version__}")
        except ImportError:
            raise ImportError("ONNX package is not installed. Please install it with 'pip install onnx'.")

        logger.info("Loading ruElectra model...")
        model_name = "sberbank-ai/ruElectra-small"
        
        # Check PyTorch version and CUDA availability (for information only)
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        # Force CPU device for conversion to ensure compatibility
        device = torch.device("cpu")
        logger.info(f"Using device: {device}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.to(device)
            model.eval()
            logger.info("Successfully loaded the model and tokenizer")
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {str(e)}")
            raise
        
        logger.info("Creating example input for tracing...")
        # Create sample inputs for tracing
        sample_text = ["Привет! Как твои дела?"]
        encoded_input = tokenizer(sample_text, padding=True, truncation=True,
                                  max_length=24, return_tensors='pt')
        
        # Move inputs to the same device as model
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        # Get input names
        input_names = list(encoded_input.keys())
        output_names = ["last_hidden_state"]

        logger.info(f"Input names: {input_names}")
        logger.info(f"Output names: {output_names}")

        # Dynamic axes for variable sequence length inference
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "last_hidden_state": {0: "batch_size", 1: "sequence"}
        }

        if "token_type_ids" in input_names:
            dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence"}

        logger.info("Exporting model to ONNX format...")
        onnx_path = "ruelectra_small.onnx"
        try:
            # Export to ONNX
            torch.onnx.export(
                model,
                tuple(encoded_input.values()),  # Model inputs as tuple of tensors
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=12,
                do_constant_folding=True,
                verbose=False
            )
            logger.info(f"Model exported to {onnx_path}")
        except Exception as e:
            logger.error(f"Error during ONNX export: {str(e)}")
            raise

        logger.info("Saving tokenizer...")
        # Create directory if it doesn't exist
        tokenizer_path = "./onnx_tokenizer"
        os.makedirs(tokenizer_path, exist_ok=True)

        # Save tokenizer configuration for later use
        tokenizer.save_pretrained(tokenizer_path)
        logger.info(f"Tokenizer saved to {tokenizer_path}")

        # Verify the ONNX model
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validation passed!")
            
            # Print model info
            logger.info(f"Model inputs: {[i.name for i in onnx_model.graph.input]}")
            logger.info(f"Model outputs: {[o.name for o in onnx_model.graph.output]}")
            
        except Exception as e:
            logger.warning(f"ONNX model validation failed: {str(e)}")

        logger.info("ONNX conversion complete!")
        return onnx_path

    except Exception as e:
        logger.error(f"Error converting model to ONNX: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    convert_electra_to_onnx()