import math

import torch
import torch.nn as nn

from typing import List, Optional

class PrepareForMultiHeadAttention(nn.Module):
    """ A module to prepare input for multi-head attention by projecting the input tensor into multiple heads with specified dimensions.
    This module is typically used in transformer architectures to split the input into multiple attention heads, allowing the model to jointly attend to information from different representation subspaces at different positions.
    Args:
        d_model (int): The dimension of the input tensor.
        n_heads (int): The number of attention heads.
        d_heads (int): The dimension of each attention head.
        bias (bool): Whether to include a bias term in the linear transformation.
    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
    Outputs:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, n_heads, d_heads).
    Example:
        >>> d_model = 512
        >>> n_heads = 8
        >>> d_heads = 64
        >>> bias = True
        >>> input_tensor = torch.randn(32, 10, d_model)  # batch_size=32, seq_len=10
        >>> mha_preparer = PrepareForMultiHeadAttention(d_model, n_heads, d_heads, bias)
        >>> output_tensor = mha_preparer(input_tensor)
        >>> output_tensor.shape
        torch.Size([32, 10, 8, 64])
    """
    def __init__(self, d_model: int, n_heads: int, d_heads: int, bias: bool):
        super().__init__()
        # Linear layer to project the input to multiple heads
        self.linear = nn.Linear(d_model, n_heads * d_heads, bias=bias)
        self.n_heads = n_heads
        self.d_heads = d_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_shape = x.shape[:-1]

        # Apply the linear transformation to the input tensor
        x = self.linear(x)
        # Reshape the output to (seq_len, batch_size, n_heads, d_heads)
        x = x.view(*head_shape, self.n_heads, self.d_heads)

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_heads = d_model // n_heads

        self.query = PrepareForMultiHeadAttention(d_model, n_heads, self.d_heads, bias)
        self.key = PrepareForMultiHeadAttention(d_model, n_heads, self.d_heads, bias)
        self.value = PrepareForMultiHeadAttention(d_model, n_heads, self.d_heads, bias=True)

        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_heads)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:


        return torch.einsum("bihd,bjhd->bijh", query, key)

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]) -> torch.Tensor:
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0], "Mask batch size must match query batch size."
        assert mask.shape[1] == 1 or mask.shape[1] == query_shape[1], "Mask sequence length must match query sequence length."
        assert mask.shape[2] == key_shape[1], "Mask sequence length must match key sequence length."
        
        mask = mask.unsqueeze(-1)
        
        return mask

    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask: Optional[torch.Tensor]) -> torch.Tensor:
        
        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
            
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        scores = self.get_scores(query, key) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = self.softmax(scores)
        attn = self.dropout(attn)

        x = torch.einsum("bijh,bjhd->bihd", attn, value)

        self.attn = attn.detach()

        x = x.reshape(seq_len, batch_size, -1)

        return self.output(x)
    
    