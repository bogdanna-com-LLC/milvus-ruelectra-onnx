import torch


def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on token embeddings, accounting for padding via attention mask.

    Args:
        model_output: Output from the transformer model
        attention_mask: Attention mask from tokenizer

    Returns:
        Mean-pooled sentence embeddings
    """
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-10)
    return sum_embeddings / sum_mask