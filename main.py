"""
Reasoning LLM with MoE and CoT - Main Entry Point

This script demonstrates how to:
1. Configure the model for your GPU (RTX 4070 8GB)
2. Create the model architecture
3. Train on synthetic or real data
4. Generate text with reasoning

Usage:
    # Test model creation
    python main.py --test
    
    # Train on synthetic data
    python main.py --train --synthetic
    
    # Train on HuggingFace dataset
    python main.py --train --dataset wikitext --subset wikitext-2-raw-v1
    
    # Generate text
    python main.py --generate --prompt "What is the capital of France?"
"""

import argparse
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from model.config import ModelConfig, CONFIGS
from model.transformer import ReasoningLLM
from training.trainer import Trainer, TrainingConfig, count_parameters, estimate_memory_usage
from training.data import (
    create_tokenizer,
    create_dataloaders,
    generate_synthetic_math_data,
    ReasoningDataset,
)


def test_model():
    """Test model creation and forward pass."""
    print("\n" + "="*60)
    print("TESTING MODEL ARCHITECTURE")
    print("="*60)
    
    # Create model with small config
    config = CONFIGS["small"]
    config.print_config()
    
    print("\nCreating model...")
    model = ReasoningLLM(config)
    
    # Count parameters
    total, trainable = count_parameters(model)
    print(f"\nParameter count:")
    print(f"  Total: {total:,} ({total/1e6:.2f}M)")
    print(f"  Trainable: {trainable:,}")
    
    # Estimate memory
    train_config = TrainingConfig()
    memory = estimate_memory_usage(
        model,
        batch_size=train_config.micro_batch_size,
        seq_len=config.max_position_embeddings,
        config=train_config,
    )
    
    print(f"\nEstimated GPU Memory:")
    print(f"  Parameters: {memory['parameters_mb']:.1f} MB")
    print(f"  Gradients: {memory['gradients_mb']:.1f} MB")
    print(f"  Optimizer: {memory['optimizer_mb']:.1f} MB")
    print(f"  Activations: {memory['activations_mb']:.1f} MB")
    print(f"  TOTAL: {memory['total_gb']:.2f} GB")
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"\nDevice: {device}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 64
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    
    # Standard forward
    outputs = model(input_ids, labels=labels)
    print(f"  Standard forward:")
    print(f"    Logits shape: {outputs['logits'].shape}")
    print(f"    Loss: {outputs['loss'].item():.4f}")
    if outputs['aux_loss']:
        print(f"    MoE aux loss: {outputs['aux_loss']:.4f}")
    
    # Forward with reasoning
    outputs_reasoning = model(input_ids, labels=labels, use_reasoning=True)
    print(f"  With reasoning:")
    print(f"    Loss: {outputs_reasoning['loss'].item():.4f}")
    if outputs_reasoning['reasoning_info']:
        print(f"    Reasoning info: {outputs_reasoning['reasoning_info']}")
    
    # Test generation
    print("\nTesting generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 10), device=device)
    generated = model.generate(prompt, max_new_tokens=30, temperature=0.8)
    print(f"  Prompt length: {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    
    # GPU memory after forward pass
    if torch.cuda.is_available():
        print(f"\nActual GPU Memory Used:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
    
    print("\n✓ Model test passed!")
    return model, config


def train_synthetic(args):
    """Train on synthetic math data."""
    print("\n" + "="*60)
    print("TRAINING ON SYNTHETIC DATA")
    print("="*60)
    
    # Create model
    config = CONFIGS["small"]
    model = ReasoningLLM(config)
    
    # Create tokenizer
    tokenizer = create_tokenizer()
    
    # Generate synthetic data
    print("\nGenerating synthetic math problems...")
    train_data = generate_synthetic_math_data(num_samples=5000)
    eval_data = generate_synthetic_math_data(num_samples=500)
    
    # Create datasets
    from torch.utils.data import DataLoader
    train_dataset = ReasoningDataset(train_data, tokenizer, max_length=256)
    eval_dataset = ReasoningDataset(eval_data, tokenizer, max_length=256)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Create dataloaders
    train_config = TrainingConfig(
        micro_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=args.max_steps,
        eval_interval=100,
        save_interval=500,
        learning_rate=1e-4,
        use_amp=True,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.micro_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=train_config.micro_batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=train_config,
    )
    
    # Train
    trainer.train()
    
    return trainer


def train_dataset(args):
    """Train on a HuggingFace dataset."""
    print("\n" + "="*60)
    print(f"TRAINING ON DATASET: {args.dataset}")
    print("="*60)
    
    # Create model
    config = CONFIGS["small"]
    model = ReasoningLLM(config)
    
    # Create tokenizer
    tokenizer = create_tokenizer()
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    
    from datasets import load_dataset
    
    if args.subset:
        dataset = load_dataset(args.dataset, args.subset)
    else:
        dataset = load_dataset(args.dataset)
    
    # Find text column
    train_data = dataset['train']
    text_columns = ['text', 'content', 'document']
    text_col = None
    for col in text_columns:
        if col in train_data.column_names:
            text_col = col
            break
    if text_col is None:
        text_col = train_data.column_names[0]
    
    print(f"Using text column: {text_col}")
    print(f"Training samples: {len(train_data)}")
    
    # Create dataset
    from training.data import TextDataset
    from torch.utils.data import DataLoader
    
    texts = train_data[text_col]
    if args.max_samples:
        texts = texts[:args.max_samples]
    
    train_dataset = TextDataset(
        texts,
        tokenizer,
        max_length=config.max_position_embeddings,
    )
    
    # Create config and dataloader
    train_config = TrainingConfig(
        micro_batch_size=4,
        gradient_accumulation_steps=8,
        max_steps=args.max_steps,
        eval_interval=200,
        save_interval=1000,
        learning_rate=3e-4,
        use_amp=True,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.micro_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    # Create trainer (no eval for simplicity)
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=None,
        config=train_config,
    )
    
    # Train
    trainer.train()
    
    return trainer


def generate_text(args):
    """Generate text with the model."""
    print("\n" + "="*60)
    print("TEXT GENERATION")
    print("="*60)
    
    # Create model
    config = CONFIGS["small"]
    model = ReasoningLLM(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create tokenizer
    tokenizer = create_tokenizer()
    
    # Tokenize prompt
    prompt = args.prompt
    print(f"\nPrompt: {prompt}")
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate
    print(f"\nGenerating (use_reasoning={args.use_reasoning})...")
    
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_reasoning=args.use_reasoning,
        )
    
    # Decode
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    
    print(f"\nGenerated text:")
    print("-" * 40)
    print(output_text)
    print("-" * 40)
    
    return output_text


def main():
    parser = argparse.ArgumentParser(
        description="Reasoning LLM with MoE and CoT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Mode selection
    parser.add_argument('--test', action='store_true',
                       help='Test model architecture')
    parser.add_argument('--train', action='store_true',
                       help='Train the model')
    parser.add_argument('--generate', action='store_true',
                       help='Generate text')
    
    # Training options
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic math data for training')
    parser.add_argument('--dataset', type=str,
                       help='HuggingFace dataset to train on')
    parser.add_argument('--subset', type=str,
                       help='Dataset subset/config name')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum training steps')
    parser.add_argument('--max-samples', type=int,
                       help='Maximum training samples to use')
    
    # Generation options
    parser.add_argument('--prompt', type=str, default="Once upon a time",
                       help='Prompt for text generation')
    parser.add_argument('--max-tokens', type=int, default=100,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Nucleus sampling threshold')
    parser.add_argument('--use-reasoning', action='store_true',
                       help='Use CoT reasoning during generation')
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint file')
    
    args = parser.parse_args()
    
    # Print GPU info
    print("\n" + "="*60)
    print("REASONING LLM WITH MOE AND COT")
    print("="*60)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {gpu_memory:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: CUDA not available, using CPU")
    
    print(f"PyTorch Version: {torch.__version__}")
    
    # Execute requested mode
    if args.test:
        test_model()
    elif args.train:
        if args.synthetic:
            train_synthetic(args)
        elif args.dataset:
            train_dataset(args)
        else:
            print("Please specify --synthetic or --dataset for training")
            parser.print_help()
    elif args.generate:
        generate_text(args)
    else:
        # Default: run test
        print("\nNo mode specified, running test...")
        test_model()


if __name__ == "__main__":
    main()
