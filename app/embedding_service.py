import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional


class RuElectraEmbeddingService:
    """
    Service for generating sentence embeddings using RuElectra model
    """

    def __init__(
        self,
        model_name: str = "sberbank-ai/ruElectra-small",
        max_length: int = 24
    ):
        """
        Initialize the embedding service

        :param model_name: Hugging Face model name
        :param max_length: Maximum token length
        """
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Store configuration
        self.max_length = max_length

    def mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on model output

        :param model_output: Model's output
        :param attention_mask: Attention mask tensor
        :return: Pooled sentence embeddings
        """
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]

        # Expand attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-10)

        return sum_embeddings / sum_mask

    def generate_embeddings(
        self,
        sentences: List[str],
        return_tensors: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for input sentences

        :param sentences: List of input sentences
        :param return_tensors: Whether to return torch tensors
        :return: List of sentence embeddings
        """
        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)

        # Compute embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = self.mean_pooling(
                model_output,
                encoded_input['attention_mask']
            )

        # Move to CPU if needed and convert to list
        if return_tensors:
            return sentence_embeddings
        else:
            return sentence_embeddings.cpu().numpy().tolist()

    def get_model_info(self):
        """
        Retrieve model and device information

        :return: Dictionary with model details
        """
        return {
            "model_name": self.model.config.model_type,
            "device": str(self.device),
            "max_length": self.max_length,
            "embedding_dim": self.model.config.hidden_size
        }