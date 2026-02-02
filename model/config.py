"""
Model Configuration for Small Reasoning LLM with MoE

This configuration is optimized for training on an RTX 4070 with 8GB VRAM.
The model is designed to be small enough to fit in memory while still 
demonstrating the key architectural concepts of modern LLMs.

Key Design Decisions:
- Small vocabulary (32K) to reduce embedding memory
- 6 layers to balance depth and memory usage
- 512 hidden dimension for reasonable expressiveness
- 8 attention heads with 64 dims each
- MoE with 4 experts, top-2 routing to increase capacity without linear compute increase
- Context length of 512 tokens for training efficiency
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """
    Configuration class for the Reasoning LLM with MoE.
    
    Memory Estimation for 8GB VRAM RTX 4070:
    - Model parameters: ~50M params × 4 bytes = ~200MB (fp32) or ~100MB (fp16)
    - Activations: Batch × Seq × Hidden × Layers ≈ variable
    - Optimizer states: 2× model size for AdamW
    - Gradients: 1× model size
    
    Total estimated: ~1-2GB for model, leaving room for batch processing
    """
    
    # Vocabulary and Embedding
    vocab_size: int = 32000  # BPE vocabulary size (similar to LLaMA)
    hidden_size: int = 512   # Model dimension (d_model)
    
    # Transformer Architecture
    num_layers: int = 6                    # Number of transformer blocks
    num_attention_heads: int = 8           # Number of attention heads
    head_dim: int = 64                     # Dimension per head (hidden_size // num_heads)
    intermediate_size: int = 1408          # FFN intermediate size (~2.75× hidden for SwiGLU)
    
    # Mixture of Experts (MoE)
    use_moe: bool = True                   # Enable MoE layers
    num_experts: int = 4                   # Total number of experts
    num_experts_per_token: int = 2         # Top-k experts activated per token
    moe_layer_frequency: int = 2           # Apply MoE every N layers (others use dense FFN)
    
    # Chain of Thought (CoT)
    use_cot: bool = True                   # Enable CoT reasoning module
    cot_hidden_size: int = 256             # Hidden size for CoT reasoning
    max_reasoning_steps: int = 4           # Maximum reasoning steps
    
    # Positional Encoding
    max_position_embeddings: int = 512     # Maximum sequence length
    rope_theta: float = 10000.0            # RoPE base frequency
    
    # Regularization
    hidden_dropout: float = 0.1            # Dropout for hidden states
    attention_dropout: float = 0.1         # Dropout for attention weights
    
    # Normalization
    rms_norm_eps: float = 1e-6             # RMSNorm epsilon
    
    # Training
    initializer_range: float = 0.02        # Weight initialization std
    use_cache: bool = True                 # Enable KV-cache for inference
    tie_word_embeddings: bool = True       # Tie input/output embeddings
    
    # Memory Optimization
    gradient_checkpointing: bool = False   # Trade compute for memory
    use_flash_attention: bool = True       # Use Flash Attention if available
    
    def __post_init__(self):
        """Validate configuration and compute derived values."""
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"
        
        # Ensure head_dim matches
        expected_head_dim = self.hidden_size // self.num_attention_heads
        if self.head_dim != expected_head_dim:
            print(f"Warning: Adjusting head_dim from {self.head_dim} to {expected_head_dim}")
            self.head_dim = expected_head_dim
    
    def estimate_parameters(self) -> dict:
        """Estimate the number of parameters in the model."""
        # Embeddings
        embedding_params = self.vocab_size * self.hidden_size
        
        # Per transformer layer
        attention_params = 4 * self.hidden_size * self.hidden_size  # Q, K, V, O projections
        
        # FFN or MoE per layer
        if self.use_moe:
            # MoE layers have multiple experts
            moe_expert_params = self.num_experts * (
                2 * self.hidden_size * self.intermediate_size +  # gate and up proj
                self.intermediate_size * self.hidden_size        # down proj
            )
            moe_router_params = self.hidden_size * self.num_experts
            dense_ffn_params = (
                2 * self.hidden_size * self.intermediate_size +
                self.intermediate_size * self.hidden_size
            )
            
            # Calculate based on MoE frequency
            moe_layers = self.num_layers // self.moe_layer_frequency
            dense_layers = self.num_layers - moe_layers
            ffn_params = (moe_layers * (moe_expert_params + moe_router_params) +
                         dense_layers * dense_ffn_params)
        else:
            ffn_params = self.num_layers * (
                2 * self.hidden_size * self.intermediate_size +
                self.intermediate_size * self.hidden_size
            )
        
        # Layer norms (2 per layer + 1 final)
        norm_params = (2 * self.num_layers + 1) * self.hidden_size
        
        # CoT module
        cot_params = 0
        if self.use_cot:
            cot_params = (
                self.hidden_size * self.cot_hidden_size +  # input projection
                self.cot_hidden_size * self.cot_hidden_size +  # reasoning GRU
                self.cot_hidden_size * self.hidden_size +  # output projection
                self.cot_hidden_size  # step embedding
            )
        
        total_params = (
            embedding_params +
            self.num_layers * attention_params +
            ffn_params +
            norm_params +
            cot_params
        )
        
        # If tying embeddings, output projection is free
        if not self.tie_word_embeddings:
            total_params += self.vocab_size * self.hidden_size
        
        return {
            "embedding": embedding_params,
            "attention": self.num_layers * attention_params,
            "ffn_moe": ffn_params,
            "norm": norm_params,
            "cot": cot_params,
            "total": total_params,
            "total_millions": total_params / 1e6,
        }
    
    def print_config(self):
        """Print configuration summary."""
        params = self.estimate_parameters()
        print("=" * 60)
        print("Reasoning LLM Configuration")
        print("=" * 60)
        print(f"Hidden Size: {self.hidden_size}")
        print(f"Num Layers: {self.num_layers}")
        print(f"Num Attention Heads: {self.num_attention_heads}")
        print(f"Head Dimension: {self.head_dim}")
        print(f"Vocabulary Size: {self.vocab_size}")
        print(f"Max Position Embeddings: {self.max_position_embeddings}")
        print("-" * 60)
        print(f"MoE Enabled: {self.use_moe}")
        if self.use_moe:
            print(f"  Num Experts: {self.num_experts}")
            print(f"  Top-K Experts: {self.num_experts_per_token}")
            print(f"  MoE Layer Frequency: Every {self.moe_layer_frequency} layers")
        print("-" * 60)
        print(f"CoT Enabled: {self.use_cot}")
        if self.use_cot:
            print(f"  CoT Hidden Size: {self.cot_hidden_size}")
            print(f"  Max Reasoning Steps: {self.max_reasoning_steps}")
        print("-" * 60)
        print("Parameter Estimates:")
        for key, value in params.items():
            if key == "total_millions":
                print(f"  Total: {value:.2f}M parameters")
            elif key != "total":
                print(f"  {key}: {value:,}")
        print("=" * 60)


# Preset configurations for different VRAM budgets
CONFIGS = {
    "tiny": ModelConfig(
        vocab_size=50257,
        hidden_size=256,
        num_layers=4,
        num_attention_heads=4,
        intermediate_size=704,
        num_experts=2,
        max_position_embeddings=256,
    ),
    "small": ModelConfig(
        vocab_size=50257,
        hidden_size=512,
        num_layers=6,
        num_attention_heads=8,
        intermediate_size=1408,
        num_experts=4,
        max_position_embeddings=512,
    ),
    "medium": ModelConfig(
        vocab_size=50257,
        hidden_size=768,
        num_layers=8,
        num_attention_heads=12,
        intermediate_size=2048,
        num_experts=8,
        max_position_embeddings=1024,
    ),
}


if __name__ == "__main__":
    # Test configuration
    config = CONFIGS["small"]
    config.print_config()
