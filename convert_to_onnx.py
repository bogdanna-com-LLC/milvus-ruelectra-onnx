# convert_to_onnx.py
import torch
from transformers import AutoTokenizer, AutoModel
import os
import sys


def convert_electra_to_onnx():
    try:
        print("Checking for required packages...")
        # Verify ONNX is installed
        try:
            import onnx
            print(f"ONNX version: {onnx.__version__}")
        except ImportError:
            raise ImportError("ONNX package is not installed. Please install it with 'pip install onnx'.")

        print("Loading ruElectra model...")
        model_name = "sberbank-ai/ruElectra-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        print("Creating example input for tracing...")
        # Create sample inputs for tracing
        sample_text = ["Привет! Как твои дела?"]
        encoded_input = tokenizer(sample_text, padding=True, truncation=True,
                                  max_length=24, return_tensors='pt')

        # Get input names
        input_names = list(encoded_input.keys())
        output_names = ["last_hidden_state"]

        print(f"Input names: {input_names}")
        print(f"Output names: {output_names}")

        # Dynamic axes for variable sequence length inference
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "last_hidden_state": {0: "batch_size", 1: "sequence"}
        }

        if "token_type_ids" in input_names:
            dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence"}

        print("Exporting model to ONNX format...")
        try:
            # Export to ONNX
            torch.onnx.export(
                model,
                tuple(encoded_input.values()),  # Model inputs as tuple of tensors
                "ruelectra_small.onnx",
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=12,
                do_constant_folding=True,
                verbose=False
            )
        except Exception as e:
            print(f"Error during ONNX export: {str(e)}")
            raise

        print("Saving tokenizer...")
        # Create directory if it doesn't exist
        os.makedirs("./onnx_tokenizer", exist_ok=True)

        # Save tokenizer configuration for later use
        tokenizer.save_pretrained("./onnx_tokenizer")

        # Verify the ONNX model
        try:
            import onnx
            onnx_model = onnx.load("ruelectra_small.onnx")
            onnx.checker.check_model(onnx_model)
            print("ONNX model is valid!")
        except Exception as e:
            print(f"Warning: ONNX model validation failed: {str(e)}")

        print("ONNX conversion complete!")
        return "ruelectra_small.onnx"

    except Exception as e:
        print(f"Error converting model to ONNX: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    convert_electra_to_onnx()