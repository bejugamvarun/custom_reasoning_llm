"""
Chain of Thought (CoT) Module

Chain of Thought enables step-by-step reasoning in language models.
Instead of directly outputting an answer, the model can:
1. Generate intermediate reasoning steps
2. Maintain a reasoning state across steps
3. Refine understanding through iterative processing

This implementation adds a lightweight reasoning module that:
- Learns to break down problems into steps
- Maintains hidden state across reasoning iterations
- Produces reasoning-enhanced representations

Key Concepts:
1. Reasoning Steps: Fixed number of internal reasoning iterations
2. State Evolution: GRU-based state update across steps
3. Step Embeddings: Position information for each reasoning step
4. Attention Integration: Reasoning state influences attention

This is a simplified version of techniques used in:
- Quiet-STaR (Self-Taught Reasoner)
- Chain-of-Thought Fine-tuning
- Reasoning-enhanced Transformers
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import ModelConfig


class ReasoningCell(nn.Module):
    """
    Single Reasoning Step Cell
    
    Implements one step of reasoning using a GRU-like update:
    1. Combine current hidden state with input
    2. Compute update and reset gates
    3. Produce new reasoning state
    
    This allows the model to iteratively refine its understanding.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # GRU-style gates
        self.input_proj = nn.Linear(input_size, hidden_size * 3, bias=False)
        self.hidden_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single reasoning step.
        
        Args:
            x: Input features [batch, seq, input_size]
            hidden: Previous reasoning state [batch, seq, hidden_size]
        
        Returns:
            New reasoning state [batch, seq, hidden_size]
        """
        # Project input and hidden state
        input_gates = self.input_proj(x)
        hidden_gates = self.hidden_proj(hidden)
        
        # Split into reset, update, and new gates
        i_r, i_u, i_n = input_gates.chunk(3, dim=-1)
        h_r, h_u, h_n = hidden_gates.chunk(3, dim=-1)
        
        # Reset gate: how much of previous state to forget
        reset_gate = torch.sigmoid(i_r + h_r)
        
        # Update gate: how much to update vs keep
        update_gate = torch.sigmoid(i_u + h_u)
        
        # New candidate state
        new_hidden = torch.tanh(i_n + reset_gate * h_n)
        
        # Interpolate between old and new
        hidden = (1 - update_gate) * hidden + update_gate * new_hidden
        
        return self.layer_norm(hidden)


class ChainOfThoughtModule(nn.Module):
    """
    Chain of Thought Reasoning Module
    
    This module performs iterative reasoning on the hidden states
    to enhance the model's ability to handle complex reasoning tasks.
    
    Architecture:
    1. Project transformer hidden states to reasoning space
    2. Perform multiple reasoning steps with state evolution
    3. Project back to transformer hidden space
    4. Optionally output reasoning trace for interpretability
    
    The reasoning happens in a lower-dimensional space (cot_hidden_size)
    to reduce computational cost while maintaining reasoning capacity.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.cot_hidden_size = config.cot_hidden_size
        self.max_steps = config.max_reasoning_steps
        
        # Project to reasoning space
        self.input_proj = nn.Linear(self.hidden_size, self.cot_hidden_size)
        
        # Reasoning cell
        self.reasoning_cell = ReasoningCell(self.cot_hidden_size, self.cot_hidden_size)
        
        # Step embeddings (to inform which reasoning step we're at)
        self.step_embeddings = nn.Embedding(self.max_steps, self.cot_hidden_size)
        
        # Project back to transformer space
        self.output_proj = nn.Linear(self.cot_hidden_size, self.hidden_size)
        
        # Gate to control how much reasoning output to use
        self.output_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid()
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        num_steps: Optional[int] = None,
        output_reasoning_trace: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Perform chain of thought reasoning.
        
        Args:
            hidden_states: [batch, seq, hidden_size]
            num_steps: Number of reasoning steps (default: max_steps)
            output_reasoning_trace: Whether to return intermediate states
        
        Returns:
            - enhanced_hidden: [batch, seq, hidden_size]
            - reasoning_trace: List of [batch, seq, cot_hidden] (optional)
        """
        batch_size, seq_len, _ = hidden_states.shape
        num_steps = num_steps or self.max_steps
        
        # Project to reasoning space
        reasoning_state = self.input_proj(hidden_states)  # [batch, seq, cot_hidden]
        
        # Store reasoning trace for interpretability
        reasoning_trace = [reasoning_state] if output_reasoning_trace else None
        
        # Perform reasoning steps
        for step in range(num_steps):
            # Get step embedding
            step_embed = self.step_embeddings(
                torch.tensor(step, device=hidden_states.device)
            )
            
            # Add step information to reasoning state
            step_input = reasoning_state + step_embed.unsqueeze(0).unsqueeze(0)
            
            # Reasoning update
            reasoning_state = self.reasoning_cell(step_input, reasoning_state)
            
            if output_reasoning_trace:
                reasoning_trace.append(reasoning_state)
        
        # Project back to transformer space
        reasoning_output = self.output_proj(reasoning_state)
        reasoning_output = self.dropout(reasoning_output)
        
        # Gated residual connection
        # This allows the model to learn when to use reasoning
        gate_input = torch.cat([hidden_states, reasoning_output], dim=-1)
        gate = self.output_gate(gate_input)
        
        # Blend original and reasoning-enhanced representations
        enhanced_hidden = hidden_states + gate * reasoning_output
        enhanced_hidden = self.layer_norm(enhanced_hidden)
        
        return enhanced_hidden, reasoning_trace


class ReasoningHead(nn.Module):
    """
    Optional Reasoning Classification Head
    
    Predicts whether the current input requires reasoning
    and how many steps might be needed.
    
    This enables adaptive reasoning depth based on input complexity.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.max_steps = config.max_reasoning_steps
        
        # Reasoning necessity classifier
        self.needs_reasoning = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Step count predictor
        self.step_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, self.max_steps),
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict reasoning requirements.
        
        Args:
            hidden_states: [batch, seq, hidden_size]
        
        Returns:
            - needs_reasoning: [batch, seq, 1] - probability of needing reasoning
            - step_logits: [batch, seq, max_steps] - logits for number of steps
        """
        # Pool across sequence for global reasoning decision
        pooled = hidden_states.mean(dim=1)  # [batch, hidden_size]
        
        needs_reasoning = self.needs_reasoning(pooled)
        step_logits = self.step_predictor(pooled)
        
        return needs_reasoning, step_logits


class AdaptiveCoT(nn.Module):
    """
    Adaptive Chain of Thought
    
    Dynamically decides:
    1. Whether to apply reasoning
    2. How many reasoning steps to use
    
    This is more efficient than always using maximum steps.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cot_module = ChainOfThoughtModule(config)
        self.reasoning_head = ReasoningHead(config)
        self.threshold = 0.5  # Reasoning threshold
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        force_reasoning: bool = False,
        force_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Adaptive reasoning forward pass.
        
        Args:
            hidden_states: [batch, seq, hidden_size]
            force_reasoning: Always apply reasoning
            force_steps: Force specific number of steps
        
        Returns:
            - output: [batch, seq, hidden_size]
            - info: Dict with reasoning statistics
        """
        # Predict reasoning requirements
        needs_reasoning_prob, step_logits = self.reasoning_head(hidden_states)
        
        # Decide whether to reason
        should_reason = force_reasoning or (needs_reasoning_prob.mean() > self.threshold)
        
        info = {
            "reasoning_prob": needs_reasoning_prob.mean().item(),
            "should_reason": should_reason,
            "predicted_steps": step_logits.argmax(dim=-1).float().mean().item() + 1,
        }
        
        if not should_reason:
            return hidden_states, info
        
        # Determine number of steps
        if force_steps is not None:
            num_steps = force_steps
        else:
            num_steps = int(step_logits.argmax(dim=-1).float().mean().item()) + 1
            num_steps = min(num_steps, self.cot_module.max_steps)
        
        info["actual_steps"] = num_steps
        
        # Apply reasoning
        enhanced_hidden, _ = self.cot_module(hidden_states, num_steps=num_steps)
        
        return enhanced_hidden, info


if __name__ == "__main__":
    # Test CoT modules
    from .config import CONFIGS
    
    config = CONFIGS["small"]
    cot = ChainOfThoughtModule(config)
    adaptive_cot = AdaptiveCoT(config)
    
    # Test input
    batch_size = 2
    seq_len = 64
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Test basic CoT
    print("Testing ChainOfThoughtModule:")
    enhanced, trace = cot(x, output_reasoning_trace=True)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {enhanced.shape}")
    print(f"  Reasoning trace steps: {len(trace)}")
    
    # Test adaptive CoT
    print("\nTesting AdaptiveCoT:")
    output, info = adaptive_cot(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Reasoning info: {info}")
