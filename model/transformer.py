"""
Transformer Architecture for Reasoning LLM

This module implements the full transformer architecture combining:
1. Multi-Head Attention with RoPE
2. Mixture of Experts (MoE) or Dense FFN
3. Chain of Thought (CoT) reasoning
4. RMSNorm for layer normalization

Architecture Overview:
┌─────────────────────────────────────┐
│           Input Embeddings          │
│         + Positional Encoding       │
└─────────────────┬───────────────────┘
                  │
          ┌───────▼───────┐
          │ Transformer   │ × N layers
          │    Block      │
          └───────┬───────┘
                  │
          ┌───────▼───────┐
          │  CoT Module   │ (optional)
          └───────┬───────┘
                  │
          ┌───────▼───────┐
          │   RMS Norm    │
          └───────┬───────┘
                  │
          ┌───────▼───────┐
          │   LM Head     │
          └───────────────┘
"""

import math
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .attention import CausalSelfAttention
from .moe import create_ffn_layer, MixtureOfExperts
from .cot import AdaptiveCoT


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    RMSNorm simplifies LayerNorm by removing the mean centering:
    RMSNorm(x) = x / RMS(x) * γ
    where RMS(x) = sqrt(mean(x²))
    
    Benefits:
    - Computationally cheaper than LayerNorm
    - Works well in practice for transformer LLMs
    - Used in LLaMA, Mistral, and other modern LLMs
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor [..., hidden_size]
        
        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        return x / rms * self.weight


class TransformerBlock(nn.Module):
    """
    Single Transformer Block
    
    Architecture (Pre-LN style):
    
    x ──┬──► RMSNorm ──► Attention ──► + ──┬──► RMSNorm ──► FFN/MoE ──► + ──► out
        │                             │    │                            │
        └─────────────────────────────┘    └────────────────────────────┘
        (residual)                         (residual)
    
    The Pre-LN architecture applies normalization before each sub-layer,
    which provides more stable training compared to Post-LN.
    """
    
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Attention sub-layer
        self.attention = CausalSelfAttention(config, layer_idx)
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # FFN or MoE sub-layer
        self.ffn = create_ffn_layer(config, layer_idx)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout)
        
        # Track if this layer uses MoE
        self.is_moe_layer = isinstance(self.ffn, MixtureOfExperts)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple], Optional[torch.Tensor]]:
        """
        Forward pass through transformer block.
        
        Args:
            hidden_states: [batch, seq, hidden_size]
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached KV for incremental decoding
            use_cache: Whether to return cache
            output_attentions: Whether to return attention weights
        
        Returns:
            - hidden_states: [batch, seq, hidden_size]
            - attention_weights: Optional attention weights
            - present_key_value: Optional updated cache
            - aux_loss: Optional MoE auxiliary loss
        """
        # ===== Attention Sub-Layer =====
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        
        attn_output, attention_weights, present_key_value = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        
        hidden_states = residual + self.dropout(attn_output)
        
        # ===== FFN/MoE Sub-Layer =====
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        
        ffn_output, router_logits, aux_loss = self.ffn(hidden_states)
        
        hidden_states = residual + self.dropout(ffn_output)
        
        return hidden_states, attention_weights, present_key_value, aux_loss


class ReasoningLLM(nn.Module):
    """
    Reasoning Language Model with MoE and CoT
    
    This is the main model class that combines all components:
    1. Token embeddings
    2. Stack of transformer blocks with MoE
    3. Chain of Thought reasoning module
    4. Language modeling head
    
    The model can operate in different modes:
    - Standard: Direct token prediction
    - Reasoning: Apply CoT before prediction for complex tasks
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Embedding dropout
        self.embed_dropout = nn.Dropout(config.hidden_dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx)
            for layer_idx in range(config.num_layers)
        ])
        
        # Final layer norm
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # Chain of Thought module
        if config.use_cot:
            self.cot_module = AdaptiveCoT(config)
        else:
            self.cot_module = None
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie embeddings if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Print model info
        self._print_model_info()
    
    def _init_weights(self, module: nn.Module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def _print_model_info(self):
        """Print model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n{'='*60}")
        print(f"Reasoning LLM initialized")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Count MoE layers
        moe_layers = sum(1 for layer in self.layers if layer.is_moe_layer)
        print(f"MoE layers: {moe_layers}/{len(self.layers)}")
        
        if self.cot_module is not None:
            print(f"CoT module: Enabled (max {self.config.max_reasoning_steps} steps)")
        
        print(f"{'='*60}\n")
    
    def get_input_embeddings(self) -> nn.Embedding:
        """Get the token embedding layer."""
        return self.token_embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_reasoning: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: [batch, seq_len] - Input token IDs
            attention_mask: [batch, seq_len] - Attention mask (1 = attend, 0 = ignore)
            position_ids: [batch, seq_len] - Position indices (optional)
            past_key_values: Cached KV tensors for incremental decoding
            use_cache: Whether to return cache for next step
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            use_reasoning: Whether to apply CoT reasoning
            labels: [batch, seq_len] - Target token IDs for loss computation
        
        Returns:
            Dictionary containing:
            - logits: [batch, seq_len, vocab_size]
            - loss: Optional cross-entropy loss
            - aux_loss: Optional MoE auxiliary loss
            - past_key_values: Optional cache
            - attentions: Optional attention weights
            - hidden_states: Optional intermediate states
            - reasoning_info: Optional CoT information
        """
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        hidden_states = self.token_embeddings(input_ids)
        hidden_states = self.embed_dropout(hidden_states)
        
        # Create position IDs if not provided
        if position_ids is None:
            if past_key_values is not None and len(past_key_values) > 0:
                past_len = past_key_values[0][0].shape[1]
            else:
                past_len = 0
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Convert attention mask to additive mask if provided
        if attention_mask is not None:
            # [batch, seq] -> [batch, 1, 1, seq]
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # Storage for outputs
        all_hidden_states = [hidden_states] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        present_key_values = [] if use_cache else None
        total_aux_loss = 0.0
        
        # Process through transformer layers
        for layer_idx, layer in enumerate(self.layers):
            # Get cached KV for this layer
            past_kv = past_key_values[layer_idx] if past_key_values is not None else None
            
            # Forward through layer
            hidden_states, attn_weights, present_kv, aux_loss = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            
            # Accumulate outputs
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            if output_attentions and attn_weights is not None:
                all_attentions.append(attn_weights)
            if use_cache:
                present_key_values.append(present_kv)
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
        
        # Apply Chain of Thought reasoning if enabled and requested
        reasoning_info = None
        if use_reasoning and self.cot_module is not None:
            hidden_states, reasoning_info = self.cot_module(hidden_states)
        
        # Final layer norm
        hidden_states = self.final_norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # Ignore padding tokens
            )
            
            # Add auxiliary MoE loss
            if total_aux_loss > 0:
                loss = loss + total_aux_loss
        
        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss if total_aux_loss > 0 else None,
            "past_key_values": present_key_values,
            "attentions": all_attentions,
            "hidden_states": all_hidden_states,
            "reasoning_info": reasoning_info,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        use_reasoning: bool = False,
        stop_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: [batch, seq_len] - Prompt token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top-k tokens for sampling
            top_p: Nucleus sampling threshold
            use_reasoning: Apply CoT for each generation step
            stop_tokens: Token IDs that stop generation
        
        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        self.eval()
        batch_size = input_ids.shape[0]
        
        # Initialize cache
        past_key_values = None
        generated = input_ids
        
        for _ in range(max_new_tokens):
            # Get context (limited by max position embeddings)
            if generated.shape[1] > self.config.max_position_embeddings:
                context = generated[:, -self.config.max_position_embeddings:]
                past_key_values = None  # Reset cache
            else:
                context = generated
            
            # If using cache, only process new token
            if past_key_values is not None:
                context = context[:, -1:]
            
            # Forward pass
            outputs = self.forward(
                context,
                past_key_values=past_key_values,
                use_cache=True,
                use_reasoning=use_reasoning,
            )
            
            logits = outputs["logits"][:, -1, :]  # [batch, vocab]
            past_key_values = outputs["past_key_values"]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for stop tokens
            if stop_tokens is not None:
                if next_token.item() in stop_tokens:
                    break
        
        return generated


if __name__ == "__main__":
    # Test the full model
    from .config import CONFIGS
    
    print("Testing ReasoningLLM...")
    
    config = CONFIGS["small"]
    model = ReasoningLLM(config)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    print(f"\nForward pass:")
    print(f"  Input shape: {input_ids.shape}")
    
    outputs = model(input_ids, labels=labels, output_attentions=True)
    
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    if outputs['aux_loss'] is not None:
        print(f"  Aux loss: {outputs['aux_loss'].item():.4f}")
    
    # Test with reasoning
    print(f"\nForward pass with reasoning:")
    outputs_reasoning = model(input_ids, labels=labels, use_reasoning=True)
    print(f"  Loss: {outputs_reasoning['loss'].item():.4f}")
    if outputs_reasoning['reasoning_info'] is not None:
        print(f"  Reasoning info: {outputs_reasoning['reasoning_info']}")
    
    # Test generation
    print(f"\nGeneration test:")
    prompt = torch.randint(0, config.vocab_size, (1, 10), device=device)
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    print(f"  Prompt length: {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
