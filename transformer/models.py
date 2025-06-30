import math
import torch
import torch.nn as nn

from .mha import MultiHeadAttention
from .feed_forward import FeedForward
from .positional_encoding import get_positional_encoding
import copy

class TransformerLayer(nn.Module):
    def __init__(self, *, 
                 d_model: int, 
                 self_attn: MultiHeadAttention, 
                 src_attn: MultiHeadAttention = None, 
                 feed_forward: FeedForward, 
                 dropout_prob: float):
        
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        # Normalize the source attention if it exists
        if self.src_attn is not None:
            self.norm_src_attn = nn.LayerNorm([d_model])
        # Normalize the feed-forward layer
        self.norm_ff = nn.LayerNorm([d_model])
        self.is_save_ff_input = False

    def forward(self, *, 
                x: torch.Tensor, 
                mask: torch.Tensor, 
                src: torch.Tensor = None, 
                src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the transformer layer.
        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :param mask: Mask tensor for self-attention.
        :param src: Source tensor for cross-attention (if applicable).
        :param src_mask: Mask tensor for source attention (if applicable).
        :return: Output tensor after processing through the layer.
        """
        # Normalize the input
        z = self.norm_self_attn(x)
        # Self-attention
        self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)
        # Apply dropout and add residual connection
        x = self.dropout(self_attn) + x

        # If source is provided, apply cross-attention (decoder case)
        if src is not None:
            z = self.norm_src_attn(x)
            # Cross-attention
            attn_src = self.src_attn(query=z, key=src, value=src, mask=src_mask)
            # Apply dropout and add residual connection
            x = self.dropout(attn_src) + x
        
        # Normalize the output before the feed-forward layer
        z = self.norm_ff(x)
        # Feed-forward layer
        if self.is_save_ff_input:
            self.ff_input = z.clone()
        ff = self.feed_forward(z)
        # Apply dropout and add residual connection
        x = self.dropout(ff) + z
        # Return the final output
        return x

class Encoder(nn.Module):
    """
    Encoder class that consists of multiple layers of encoders.
    Each layer processes the input and returns the embedded representation.
    """
    def __init__(self, 
                 layer: TransformerLayer, 
                 n_layers: int):
        
        super().__init__()
        # Make copies of the transformer layer
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])
    
    def forward(self,
                x: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        
        for layer in self.layers:
            x = layer(x=x, mask=mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    """
    Decoder class that consists of multiple layers of decoders.
    Each layer processes the input and returns the embedded representation.
    """
    def __init__(self, 
                 layer: TransformerLayer, 
                 n_layers: int):
        
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        self.norm = nn.LayerNorm([layer.size])
    
    def forward(self,
                x: torch.Tensor, 
                memory: torch.Tensor, 
                src_mask: torch.Tensor, 
                tgt_mask: torch.Tensor) -> torch.Tensor:
        
        # Process each layer in the decoder
        # memory is the output from the encoder
        for layer in self.layers:
            x = layer(x=x, mask=tgt_mask, src=memory, src_mask=src_mask)
        # Normalize the final output
        return self.norm(x)
    
class EncoderDecoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: nn.Module, tgt_embed: nn.Module, generator: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        # Encode the source sequence
        memory = self.encode(src, src_mask)
        # Decode the target sequence
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        return output
    
    def encode(self, src, src_mask):
        # Embed the source sequence
        src_embedded = self.src_embed(src)
        # Pass through the encoder
        return self.encoder(src_embedded, src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        # Embed the target sequence
        tgt_embedded = self.tgt_embed(tgt)
        # Pass through the decoder
        return self.decoder(tgt_embedded, memory, src_mask, tgt_mask)

class Generator(nn.Module):
    """
    This predicts the tokens and gives the lof softmax of those.
    You don't need this if you are using `nn.CrossEntropyLoss`.
    """

    def __init__(self, d_model: int, n_vocab: int):
        super().__init__()
        self.projection = nn.Linear(d_model, n_vocab)

    def forward(self, x):
        return self.projection(x)

class EmbeddingsWithPositionalEncoding(nn.Module):
    """
    Embed tokens and add [fixed positional encoding]
    """

    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
        super().__init__()
        self.lut = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len))

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:,:x.shape[1],:].requires_grad_(False)
        return self.lut(x) * math.sqrt(self.d_model) + pe
    
class EmbeddingsWithLearnedPositionalEncoding(nn.Module):
    """
    ## Embed tokens and add parameterized positional encodings
    """

    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
        super().__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]]
        return self.linear(x) * math.sqrt(self.d_model) + pe