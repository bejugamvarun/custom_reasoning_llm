"""
Training Module for Reasoning LLM

This module provides:
1. Training loop with gradient accumulation
2. Memory-efficient training for 8GB VRAM
3. Learning rate scheduling
4. Logging and checkpointing
5. Mixed precision training (FP16/BF16)

Memory Optimization Strategies:
- Gradient accumulation: Process larger effective batches
- Mixed precision: Reduce memory by 50%
- Gradient checkpointing: Trade compute for memory
- Efficient data loading: Minimize memory fragmentation
"""

import os
import time
import math
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class TrainingConfig:
    """
    Training configuration optimized for RTX 4070 (8GB VRAM).
    
    Key settings for memory efficiency:
    - Small micro batch size (2-4)
    - Gradient accumulation for larger effective batch
    - Mixed precision training
    - Gradient clipping for stability
    """
    
    # Batch sizes
    micro_batch_size: int = 4           # Batch size per forward pass
    gradient_accumulation_steps: int = 8  # Accumulate gradients over N steps
    # Effective batch size = micro_batch_size × gradient_accumulation_steps = 32
    
    # Training duration
    max_steps: int = 10000              # Total training steps
    warmup_steps: int = 500             # LR warmup steps
    eval_interval: int = 500            # Evaluate every N steps
    save_interval: int = 1000           # Save checkpoint every N steps
    log_interval: int = 10              # Log metrics every N steps
    
    # Optimizer settings
    learning_rate: float = 3e-4         # Peak learning rate
    weight_decay: float = 0.1           # L2 regularization
    beta1: float = 0.9                  # AdamW beta1
    beta2: float = 0.95                 # AdamW beta2
    eps: float = 1e-8                   # AdamW epsilon
    max_grad_norm: float = 1.0          # Gradient clipping threshold
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"        # "cosine" or "linear"
    min_lr_ratio: float = 0.1           # Minimum LR as fraction of max
    
    # Mixed precision
    use_amp: bool = True                # Use automatic mixed precision
    amp_dtype: str = "bfloat16"          # "float16" or "bfloat16"
    
    # Memory optimization
    gradient_checkpointing: bool = False  # Trade compute for memory
    
    # Checkpointing
    output_dir: str = "checkpoints"
    resume_from: Optional[str] = None
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "reasoning-llm"
    wandb_run_name: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    
    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


class LRScheduler:
    """
    Learning rate scheduler with warmup and decay.
    
    Supports:
    - Linear warmup
    - Cosine annealing
    - Linear decay
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
    ):
        self.optimizer = optimizer
        self.config = config
        self.base_lr = config.learning_rate
        self.min_lr = config.learning_rate * config.min_lr_ratio
        self.warmup_steps = config.warmup_steps
        self.max_steps = config.max_steps
        self.current_step = 0
    
    def get_lr(self, step: int) -> float:
        """Calculate learning rate for given step."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * step / self.warmup_steps
        
        # Decay phase
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        if self.config.lr_scheduler == "cosine":
            # Cosine annealing
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        else:
            # Linear decay
            return self.base_lr - (self.base_lr - self.min_lr) * progress
    
    def step(self, step: Optional[int] = None):
        """Update learning rate."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        lr = self.get_lr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class Trainer:
    """
    Training loop for Reasoning LLM.
    
    Features:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpointing
    - Wandb logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        config: TrainingConfig,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = LRScheduler(self.optimizer, config)
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        self.amp_dtype = (
            torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
        )
        
        # Setup gradient checkpointing
        if config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
        # Setup wandb
        if config.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer with weight decay handling.
        
        We don't apply weight decay to:
        - Bias terms
        - Layer normalization weights
        - Embeddings (optional)
        """
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # No weight decay for biases and layer norms
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        # This would be implemented per-layer
        # For now, we set a flag
        if hasattr(self.model, 'config'):
            self.model.config.gradient_checkpointing = True
        print("Gradient checkpointing enabled")
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config={
                "training": vars(self.config),
                "model": vars(self.model.config) if hasattr(self.model, 'config') else {},
            },
        )
        wandb.watch(self.model, log="gradients", log_freq=100)
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Single training step (forward + backward).
        
        Args:
            batch: Dictionary with 'input_ids' and optionally 'labels'
        
        Returns:
            Dictionary with loss values
        """
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass with mixed precision
        if self.config.use_amp:
            with autocast(device_type="cuda", dtype=self.amp_dtype):
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs['loss']
                aux_loss = outputs.get('aux_loss', 0.0)
        else:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs['loss']
            aux_loss = outputs.get('aux_loss', 0.0)
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.config.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'aux_loss': aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
        }
    
    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        # Unscale gradients for clipping
        if self.config.use_amp:
            self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        
        # Optimizer step
        if self.config.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad(set_to_none=True)
        
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on eval dataset."""
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_aux_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch.get('labels', input_ids).to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            if self.config.use_amp:
                with autocast(device_type="cuda", dtype=self.amp_dtype):
                    outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            else:
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            
            total_loss += outputs['loss'].item()
            if outputs.get('aux_loss') is not None:
                total_aux_loss += outputs['aux_loss'].item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_aux_loss = total_aux_loss / num_batches if total_aux_loss > 0 else 0.0
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        
        return {
            'eval_loss': avg_loss,
            'eval_aux_loss': avg_aux_loss,
            'eval_perplexity': perplexity,
        }
    
    def save_checkpoint(self, path: Optional[str] = None, is_best: bool = False):
        """Save model checkpoint."""
        if path is None:
            path = self.output_dir / f"checkpoint-{self.global_step}"
        else:
            path = Path(path)
        
        path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_step': self.scheduler.current_step,
            'global_step': self.global_step,
            'config': self.config,
            'best_eval_loss': self.best_eval_loss,
        }
        
        if self.config.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path / "checkpoint.pt")
        
        # Save model config
        if hasattr(self.model, 'config'):
            import json
            with open(path / "model_config.json", 'w') as f:
                json.dump(vars(self.model.config), f, indent=2)
        
        print(f"Checkpoint saved to {path}")
        
        if is_best:
            best_path = self.output_dir / "best"
            if best_path.exists():
                import shutil
                shutil.rmtree(best_path)
            import shutil
            shutil.copytree(path, best_path)
            print(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path / "checkpoint.pt", map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.current_step = checkpoint['scheduler_step']
        self.global_step = checkpoint['global_step']
        self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
        
        if self.config.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Checkpoint loaded from {path} (step {self.global_step})")
    
    def train(self):
        """
        Main training loop.
        
        This implements the full training procedure:
        1. Iterate over data with gradient accumulation
        2. Update learning rate
        3. Log metrics
        4. Evaluate periodically
        5. Save checkpoints
        """
        print(f"\n{'='*60}")
        print(f"Starting training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Effective batch size: {self.config.effective_batch_size}")
        print(f"Max steps: {self.config.max_steps}")
        print(f"Mixed precision: {self.config.use_amp} ({self.config.amp_dtype})")
        print(f"{'='*60}\n")
        
        # Resume from checkpoint if specified
        if self.config.resume_from is not None:
            self.load_checkpoint(self.config.resume_from)
        
        # Training loop
        self.model.train()
        train_iter = iter(self.train_dataloader)
        
        accumulated_loss = 0.0
        accumulated_aux_loss = 0.0
        
        pbar = tqdm(
            range(self.global_step, self.config.max_steps),
            desc="Training",
            initial=self.global_step,
            total=self.config.max_steps,
        )
        
        for step in pbar:
            self.global_step = step
            
            # Accumulate gradients
            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_dataloader)
                    batch = next(train_iter)
                
                losses = self.train_step(batch)
                accumulated_loss += losses['loss']
                accumulated_aux_loss += losses['aux_loss']
            
            # Optimizer step
            grad_norm = self.optimizer_step()
            
            # Update learning rate
            lr = self.scheduler.step(step)
            
            # Average accumulated losses
            avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
            avg_aux_loss = accumulated_aux_loss / self.config.gradient_accumulation_steps
            accumulated_loss = 0.0
            accumulated_aux_loss = 0.0
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{lr:.2e}',
                'grad': f'{grad_norm:.2f}',
            })
            
            # Log metrics
            if step % self.config.log_interval == 0:
                metrics = {
                    'train/loss': avg_loss,
                    'train/aux_loss': avg_aux_loss,
                    'train/learning_rate': lr,
                    'train/grad_norm': grad_norm,
                    'train/step': step,
                }
                
                if self.config.use_wandb and WANDB_AVAILABLE:
                    wandb.log(metrics, step=step)
                
                # Log GPU memory
                if torch.cuda.is_available():
                    metrics['train/gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024**2
            
            # Evaluate
            if step > 0 and step % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                
                if eval_metrics:
                    print(f"\nStep {step}: eval_loss={eval_metrics['eval_loss']:.4f}, "
                          f"perplexity={eval_metrics['eval_perplexity']:.2f}")
                    
                    if self.config.use_wandb and WANDB_AVAILABLE:
                        wandb.log(eval_metrics, step=step)
                    
                    # Save best model
                    if eval_metrics['eval_loss'] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics['eval_loss']
                        self.save_checkpoint(is_best=True)
            
            # Save checkpoint
            if step > 0 and step % self.config.save_interval == 0:
                self.save_checkpoint()
        
        # Final save
        self.save_checkpoint()
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        print("\nTraining complete!")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_memory_usage(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    config: TrainingConfig,
) -> Dict[str, float]:
    """
    Estimate GPU memory usage for training.
    
    Components:
    1. Model parameters (fp32/fp16)
    2. Gradients (same as params)
    3. Optimizer states (2× for AdamW)
    4. Activations (varies with batch/seq)
    """
    total_params, _ = count_parameters(model)
    
    # Bytes per parameter
    param_bytes = 2 if config.use_amp else 4  # fp16 vs fp32
    grad_bytes = 4  # Always fp32 for accumulation
    optimizer_bytes = 8  # AdamW stores m and v
    
    # Memory components (in MB)
    param_memory = total_params * param_bytes / 1024**2
    grad_memory = total_params * grad_bytes / 1024**2
    optimizer_memory = total_params * optimizer_bytes / 1024**2
    
    # Rough activation estimate (varies greatly)
    # Approximate: batch × seq × hidden × num_layers × overhead
    hidden_size = model.config.hidden_size if hasattr(model, 'config') else 512
    num_layers = model.config.num_layers if hasattr(model, 'config') else 6
    activation_memory = (
        batch_size * seq_len * hidden_size * num_layers * param_bytes * 4
    ) / 1024**2
    
    total = param_memory + grad_memory + optimizer_memory + activation_memory
    
    return {
        'parameters_mb': param_memory,
        'gradients_mb': grad_memory,
        'optimizer_mb': optimizer_memory,
        'activations_mb': activation_memory,
        'total_mb': total,
        'total_gb': total / 1024,
    }


if __name__ == "__main__":
    # Test training utilities
    from model.config import CONFIGS
    from model.transformer import ReasoningLLM
    
    config = CONFIGS["small"]
    model = ReasoningLLM(config)
    
    # Count parameters
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,} ({total/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable:,}")
    
    # Estimate memory
    train_config = TrainingConfig()
    memory = estimate_memory_usage(
        model,
        batch_size=train_config.micro_batch_size,
        seq_len=config.max_position_embeddings,
        config=train_config,
    )
    
    print(f"\nEstimated Memory Usage:")
    for key, value in memory.items():
        print(f"  {key}: {value:.2f}")
