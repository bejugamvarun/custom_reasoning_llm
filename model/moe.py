"""
Mixture of Experts (MoE) Layer

MoE increases model capacity without proportionally increasing compute.
Instead of one large FFN, we have multiple "expert" FFNs, and a router
selects which experts to use for each token.

Key Concepts:
1. Router: Learns to route tokens to appropriate experts
2. Experts: Independent FFN networks, each specializing in different patterns
3. Load Balancing: Auxiliary loss to prevent router collapse (all tokens to one expert)
4. Sparse Activation: Only top-k experts are activated per token

Benefits:
- Increased model capacity without linear compute increase
- Experts can specialize in different types of inputs
- Efficient inference (only compute selected experts)

Challenges:
- Load balancing to ensure all experts are used
- Potential for expert collapse
- Communication overhead in distributed settings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import ModelConfig


class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function
    
    SwiGLU combines Swish activation with Gated Linear Units:
    SwiGLU(x) = (x * W_gate) * swish(x * W_up)
    
    where swish(x) = x * sigmoid(x)
    
    This is used in LLaMA and other modern LLMs as it provides
    better performance than ReLU or GELU.
    """
    
    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return F.silu(gate) * x  # silu = swish


class Expert(nn.Module):
    """
    Single Expert Network (Feed-Forward Network)
    
    Architecture: SwiGLU-based FFN
    - Gate projection: hidden_size -> intermediate_size
    - Up projection: hidden_size -> intermediate_size  
    - Down projection: intermediate_size -> hidden_size
    
    The intermediate_size is typically 2.67× hidden_size for SwiGLU
    (compared to 4× for ReLU) to maintain similar parameter count.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # SwiGLU projections
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        self.activation = SwiGLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert.
        
        Args:
            x: [batch * seq, hidden_size] or [batch, seq, hidden_size]
        
        Returns:
            Output of same shape as input
        """
        # Compute gate and up projections
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # Apply SwiGLU activation
        hidden = self.activation(up, gate)
        
        # Down projection
        return self.down_proj(hidden)


class Router(nn.Module):
    """
    Token-to-Expert Router
    
    The router learns to assign tokens to experts. For each token,
    it produces a probability distribution over experts.
    
    Routing Strategies:
    1. Top-k: Select top-k experts with highest probability (used here)
    2. Expert Choice: Experts choose their top tokens
    3. Hash-based: Deterministic routing based on token hash
    
    Load Balancing:
    Without balancing, the router may collapse to always selecting
    the same expert. We add an auxiliary loss to encourage uniform
    expert utilization.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.hidden_size = config.hidden_size
        
        # Router linear layer
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        
        # Noise for exploration during training (helps load balancing)
        self.noise_std = 0.1
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            hidden_states: [batch, seq, hidden] or [batch * seq, hidden]
            training: Whether in training mode (adds noise)
        
        Returns:
            - router_weights: [num_tokens, num_experts_per_token] - normalized weights
            - selected_experts: [num_tokens, num_experts_per_token] - expert indices
            - router_logits: [num_tokens, num_experts] - raw router outputs
        """
        original_shape = hidden_states.shape
        
        # Flatten to [num_tokens, hidden]
        if len(original_shape) == 3:
            hidden_states = hidden_states.view(-1, self.hidden_size)
        
        # Compute router logits
        router_logits = self.gate(hidden_states)  # [num_tokens, num_experts]
        
        # Add noise during training for exploration
        if training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise
        
        # Compute routing probabilities
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        router_weights, selected_experts = torch.topk(
            routing_probs, self.num_experts_per_token, dim=-1
        )
        
        # Normalize weights to sum to 1
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        
        return router_weights, selected_experts, router_logits


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts Layer
    
    Replaces the dense FFN in transformer blocks. Contains:
    1. Multiple expert networks
    2. A router to select experts
    3. Load balancing loss computation
    
    Forward Pass:
    1. Router assigns each token to top-k experts
    2. Each selected expert processes the token
    3. Expert outputs are weighted and summed
    
    Load Balancing Loss:
    We encourage uniform expert utilization with auxiliary loss:
    L_aux = α * num_experts * Σ(f_i * P_i)
    where f_i = fraction of tokens routed to expert i
          P_i = average routing probability for expert i
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.hidden_size = config.hidden_size
        
        # Create expert networks
        self.experts = nn.ModuleList([
            Expert(config) for _ in range(self.num_experts)
        ])
        
        # Router
        self.router = Router(config)
        
        # Load balancing coefficient
        self.load_balance_coef = 0.01
    
    def compute_load_balancing_loss(
        self,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss.
        
        This loss encourages the router to distribute tokens evenly
        across all experts, preventing expert collapse.
        
        Args:
            router_logits: [num_tokens, num_experts]
            selected_experts: [num_tokens, num_experts_per_token]
        
        Returns:
            Scalar load balancing loss
        """
        num_tokens = router_logits.shape[0]
        
        # Compute routing probabilities
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # f_i: Fraction of tokens assigned to each expert
        # Count how many tokens selected each expert
        expert_mask = F.one_hot(selected_experts, self.num_experts).float()
        expert_mask = expert_mask.sum(dim=1)  # [num_tokens, num_experts]
        tokens_per_expert = expert_mask.sum(dim=0)  # [num_experts]
        f = tokens_per_expert / (num_tokens * self.num_experts_per_token)
        
        # P_i: Average routing probability for each expert
        P = routing_probs.mean(dim=0)
        
        # Load balancing loss
        aux_loss = self.num_experts * (f * P).sum()
        
        return self.load_balance_coef * aux_loss
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through MoE layer.
        
        Args:
            hidden_states: [batch, seq, hidden]
            output_router_logits: Whether to return router logits
        
        Returns:
            - output: [batch, seq, hidden]
            - router_logits: [batch * seq, num_experts] (optional)
            - aux_loss: Scalar load balancing loss (optional)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Flatten for routing
        hidden_flat = hidden_states.view(-1, hidden_size)  # [batch * seq, hidden]
        
        # Route tokens to experts
        router_weights, selected_experts, router_logits = self.router(
            hidden_flat, training=self.training
        )
        
        # Initialize output
        final_output = torch.zeros_like(hidden_flat)
        
        # Process each expert
        # This can be parallelized, but sequential is clearer for understanding
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            
            # Find tokens assigned to this expert
            # selected_experts: [num_tokens, top_k]
            expert_mask = (selected_experts == expert_idx)  # [num_tokens, top_k]
            
            if not expert_mask.any():
                continue
            
            # Get indices of tokens using this expert and their positions in top-k
            token_indices, topk_positions = torch.where(expert_mask)
            
            if len(token_indices) == 0:
                continue
            
            # Get the tokens for this expert
            expert_input = hidden_flat[token_indices]
            
            # Process through expert
            expert_output = expert(expert_input)
            
            # Get weights for these tokens
            weights = router_weights[token_indices, topk_positions].unsqueeze(-1)
            
            # Accumulate weighted expert output
            final_output.index_add_(
                0, token_indices, expert_output * weights
            )
        
        # Reshape output
        output = final_output.view(batch_size, seq_len, hidden_size)
        
        # Compute load balancing loss during training
        aux_loss = None
        if self.training:
            aux_loss = self.compute_load_balancing_loss(router_logits, selected_experts)
        
        if output_router_logits:
            return output, router_logits, aux_loss
        
        return output, None, aux_loss


class DenseFFN(nn.Module):
    """
    Dense Feed-Forward Network (non-MoE)
    
    Used in layers that don't have MoE (based on moe_layer_frequency).
    Same architecture as a single expert: SwiGLU-based FFN.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.expert = Expert(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, None]:
        """Forward pass through dense FFN."""
        return self.expert(hidden_states), None, None


def create_ffn_layer(config: ModelConfig, layer_idx: int) -> nn.Module:
    """
    Factory function to create FFN or MoE layer based on configuration.
    
    MoE layers are placed according to moe_layer_frequency:
    - frequency=1: Every layer has MoE
    - frequency=2: Every other layer has MoE
    - etc.
    
    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer (0-indexed)
    
    Returns:
        MixtureOfExperts or DenseFFN module
    """
    if config.use_moe and (layer_idx + 1) % config.moe_layer_frequency == 0:
        return MixtureOfExperts(config)
    else:
        return DenseFFN(config)


if __name__ == "__main__":
    # Test MoE module
    from .config import CONFIGS
    
    config = CONFIGS["small"]
    moe = MixtureOfExperts(config)
    
    # Test input
    batch_size = 2
    seq_len = 64
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Training forward
    moe.train()
    output, router_logits, aux_loss = moe(x, output_router_logits=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Router logits shape: {router_logits.shape}")
    print(f"Auxiliary loss: {aux_loss.item():.4f}")
    
    # Inference forward
    moe.eval()
    with torch.no_grad():
        output, _, _ = moe(x)
    print(f"Inference output shape: {output.shape}")
