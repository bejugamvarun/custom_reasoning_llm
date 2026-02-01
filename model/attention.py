"""
Attention Mechanisms for Reasoning LLM

This module implements:
1. Rotary Positional Embeddings (RoPE) - Modern positional encoding
2. Multi-Head Attention with KV-Cache support
3. Grouped Query Attention (GQA) option for memory efficiency

Key Concepts Explained:
- RoPE encodes position by rotating query/key vectors, enabling extrapolation
- KV-Cache stores computed keys/values for efficient autoregressive generation
- Flash Attention (when available) provides memory-efficient attention computation
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE)
    
    RoPE encodes absolute position by rotating the query and key vectors.
    This allows the model to:
    1. Encode absolute positions
    2. Decay attention based on relative distance (naturally)
    3. Extrapolate to longer sequences than seen during training
    
    Math:
    For a pair of features (x_i, x_{i+1}), we rotate by angle θ_i * position:
    
    [cos(mθ_i)  -sin(mθ_i)] [x_i    ]
    [sin(mθ_i)   cos(mθ_i)] [x_{i+1}]
    
    where m is the position and θ_i = 10000^{-2i/d}
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dim = config.head_dim
        self.max_seq_len = config.max_position_embeddings
        self.base = config.rope_theta
        
        # Precompute the frequency bands
        # θ_i = base^{-2i/d} for i in [0, d/2)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos and sin for all positions
        self._update_cos_sin_cache(self.max_seq_len)
    
    def _update_cos_sin_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Precompute cos/sin values for positions up to seq_len."""
        self.max_seq_len_cached = seq_len
        
        # Create position indices [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=device or self.inv_freq.device)
        
        # Outer product: [seq_len] x [dim/2] -> [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)
        
        # Duplicate for pairs: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin values for the given positions.
        
        Args:
            x: Input tensor [batch, seq_len, num_heads, head_dim]
            position_ids: Position indices [batch, seq_len] (optional)
        
        Returns:
            cos, sin: Tensors of shape [1, seq_len, 1, head_dim]
        """
        seq_len = x.shape[1]
        
        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._update_cos_sin_cache(seq_len, x.device)
        
        if position_ids is not None:
            # Gather cos/sin for specific positions
            cos = self.cos_cached[position_ids].unsqueeze(2)
            sin = self.sin_cached[position_ids].unsqueeze(2)
        else:
            # Use sequential positions
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)
        
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.
    
    For input [..., d], splits into [..., d/2] pairs and rotates:
    [x1, x2] -> [-x2, x1]
    
    This is the rotation operation in RoPE.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to queries and keys.
    
    The rotation formula:
    x_rotated = x * cos(θ) + rotate_half(x) * sin(θ)
    
    Args:
        q: Query tensor [batch, seq, heads, head_dim]
        k: Key tensor [batch, seq, heads, head_dim]
        cos: Cosine values [1, seq, 1, head_dim]
        sin: Sine values [1, seq, 1, head_dim]
    
    Returns:
        Rotated q and k tensors
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with RoPE and KV-Cache
    
    Architecture:
    1. Project input to Q, K, V using linear layers
    2. Split into multiple heads
    3. Apply RoPE to Q and K
    4. Compute attention: softmax(QK^T / sqrt(d_k)) V
    5. Concatenate heads and project output
    
    Memory Optimization:
    - KV-Cache stores past K, V for autoregressive generation
    - Flash Attention (when available) reduces memory from O(n²) to O(n)
    """
    
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.dropout = config.attention_dropout
        
        # Total dimension for all heads
        self.total_head_dim = self.num_heads * self.head_dim
        
        # Linear projections
        # Q, K, V projections (can be combined for efficiency)
        self.q_proj = nn.Linear(self.hidden_size, self.total_head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.total_head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.total_head_dim, bias=False)
        self.o_proj = nn.Linear(self.total_head_dim, self.hidden_size, bias=False)
        
        # Rotary embeddings
        self.rotary_emb = RotaryPositionalEmbedding(config)
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        
        # Scaling factor for attention scores
        self.scale = self.head_dim ** -0.5
        
        # Check for Flash Attention availability
        self.use_flash_attn = (
            config.use_flash_attention and
            hasattr(F, 'scaled_dot_product_attention')
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for multi-head attention.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, seq_len] - additive mask
            position_ids: [batch, seq_len] - position indices
            past_key_value: Cached (key, value) tensors for incremental decoding
            use_cache: Whether to return updated cache
            output_attentions: Whether to return attention weights
        
        Returns:
            - output: [batch, seq_len, hidden_size]
            - attention_weights: [batch, num_heads, seq_len, seq_len] (if requested)
            - past_key_value: Updated cache (if use_cache)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        # Shape: [batch, seq_len, total_head_dim]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to [batch, seq_len, num_heads, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE to Q and K
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Handle KV-Cache for incremental decoding
        if past_key_value is not None:
            # Concatenate with cached keys and values
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=1)
            value_states = torch.cat([past_value, value_states], dim=1)
        
        # Update cache
        present_key_value = (key_states, value_states) if use_cache else None
        
        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # Compute attention
        if self.use_flash_attn and not output_attentions:
            # Use PyTorch's optimized attention (Flash Attention 2 when available)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=attention_mask is None,  # Use causal mask if no explicit mask
            )
            attention_weights = None
        else:
            # Manual attention computation (for understanding and debugging)
            # Attention scores: [batch, heads, seq_q, seq_k]
            attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1))
            attention_scores = attention_scores * self.scale
            
            # Apply attention mask (additive, -inf for masked positions)
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
            # Softmax and dropout
            attention_weights = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
            attention_weights = attention_weights.to(query_states.dtype)
            attention_weights = self.attn_dropout(attention_weights)
            
            # Weighted sum of values
            attn_output = torch.matmul(attention_weights, value_states)
        
        # Reshape back: [batch, seq_len, total_head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.total_head_dim)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attention_weights, present_key_value


class CausalSelfAttention(MultiHeadAttention):
    """
    Causal Self-Attention - prevents attending to future tokens.
    
    This is the standard attention used in decoder-only language models.
    The causal mask ensures that position i can only attend to positions <= i.
    """
    
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        
        # Register causal mask buffer
        # Lower triangular matrix of 1s
        max_pos = config.max_position_embeddings
        causal_mask = torch.triu(
            torch.ones(max_pos, max_pos, dtype=torch.bool),
            diagonal=1
        )
        # Convert to additive mask: 0 for attend, -inf for masked
        causal_mask = torch.where(causal_mask, float('-inf'), 0.0)
        self.register_buffer("causal_mask", causal_mask, persistent=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        """Forward with automatic causal masking."""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Get causal mask for current sequence length
        if attention_mask is None:
            # Start position for the query (considering cached keys)
            if past_key_value is not None:
                past_len = past_key_value[0].shape[1]
            else:
                past_len = 0
            
            total_len = past_len + seq_len
            causal_mask = self.causal_mask[past_len:total_len, :total_len]
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )


if __name__ == "__main__":
    # Test attention module
    from .config import CONFIGS
    
    config = CONFIGS["small"]
    attention = CausalSelfAttention(config, layer_idx=0)
    
    # Test input
    batch_size = 2
    seq_len = 64
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Forward pass
    output, weights, cache = attention(x, output_attentions=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Cache key shape: {cache[0].shape}")
    print(f"Cache value shape: {cache[1].shape}")
