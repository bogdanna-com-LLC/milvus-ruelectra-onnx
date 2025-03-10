import torch
from transformers import AutoTokenizer, AutoModel
from utils import mean_pooling


class EmbeddingModel:
    def __init__(self):
        # Check if CUDA is available and set the device accordingly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        model_name = "sberbank-ai/ruElectra-small"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Set model to evaluation mode
        self.model.eval()

    def generate_embeddings(self, texts, max_length=24):
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of strings to embed
            max_length: Maximum token length for truncation

        Returns:
            Tensor of embeddings, one per input text
        """
        # Tokenize the texts
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # Move inputs to the same device as the model
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        # Compute token embeddings without gradient calculation
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform mean pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Return embeddings as numpy array for easier serialization
        return sentence_embeddings.cpu().numpy()


# Create a singleton instance for efficient use
embedding_model = None


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = EmbeddingModel()
    return embedding_model