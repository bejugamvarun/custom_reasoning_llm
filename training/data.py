"""
Data Loading and Processing Utilities

This module provides:
1. Text tokenization using HuggingFace tokenizers
2. Dataset loading from various sources
3. Efficient data collation for training
4. Reasoning-specific data augmentation

Supported Data Formats:
- Plain text files
- JSON/JSONL files
- HuggingFace datasets
- Custom reasoning datasets with CoT annotations
"""

import os
import json
import random
from typing import Optional, List, Dict, Any, Iterator
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer

# Try to import datasets, but allow fallback
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


class TextDataset(Dataset):
    """
    Simple text dataset for language modeling.
    
    Processes text into fixed-length chunks for training.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        stride: int = 256,
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            stride: Overlap between consecutive chunks
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Tokenize all texts and create chunks
        self.chunks = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Create overlapping chunks
            for i in range(0, len(tokens), stride):
                chunk = tokens[i:i + max_length]
                if len(chunk) >= 32:  # Minimum chunk size
                    self.chunks.append(chunk)
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.chunks[idx]
        
        # Pad if needed
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        
        input_ids = torch.tensor(tokens[:self.max_length], dtype=torch.long)
        
        # For LM, labels are same as inputs (shifted internally)
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
            'attention_mask': (input_ids != self.tokenizer.pad_token_id).long(),
        }


class ReasoningDataset(Dataset):
    """
    Dataset for reasoning tasks with Chain of Thought.
    
    Expects data in format:
    {
        "question": "What is 2+2?",
        "reasoning": ["First, we have 2", "Adding another 2", "We get 4"],
        "answer": "4"
    }
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        include_reasoning: bool = True,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_reasoning = include_reasoning
        
        # Special tokens for reasoning
        self.question_prefix = "Question: "
        self.reasoning_prefix = "Let's think step by step:\n"
        self.answer_prefix = "Answer: "
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Build the text
        parts = [self.question_prefix + item['question']]
        
        if self.include_reasoning and 'reasoning' in item:
            reasoning_steps = item['reasoning']
            if isinstance(reasoning_steps, list):
                reasoning_text = '\n'.join(f"Step {i+1}: {step}" 
                                          for i, step in enumerate(reasoning_steps))
            else:
                reasoning_text = reasoning_steps
            parts.append(self.reasoning_prefix + reasoning_text)
        
        parts.append(self.answer_prefix + str(item['answer']))
        
        text = '\n\n'.join(parts)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }


class StreamingTextDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for large text corpora.
    
    Reads data lazily, suitable for datasets that don't fit in memory.
    """
    
    def __init__(
        self,
        file_paths: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        shuffle_files: bool = True,
    ):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_files = shuffle_files
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        files = self.file_paths.copy()
        if self.shuffle_files:
            random.shuffle(files)
        
        buffer = []
        buffer_size = self.max_length * 10  # Accumulate tokens
        
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    tokens = self.tokenizer.encode(line, add_special_tokens=False)
                    buffer.extend(tokens)
                    
                    # Yield chunks when buffer is large enough
                    while len(buffer) >= self.max_length:
                        chunk = buffer[:self.max_length]
                        buffer = buffer[self.max_length // 2:]  # Keep overlap
                        
                        input_ids = torch.tensor(chunk, dtype=torch.long)
                        yield {
                            'input_ids': input_ids,
                            'labels': input_ids.clone(),
                            'attention_mask': torch.ones_like(input_ids),
                        }


def create_tokenizer(
    vocab_size: int = 32000,
    model_name: Optional[str] = None,
) -> PreTrainedTokenizer:
    """
    Create or load a tokenizer.
    
    Args:
        vocab_size: Vocabulary size (ignored if model_name provided)
        model_name: HuggingFace model name to load tokenizer from
    
    Returns:
        Tokenizer instance
    """
    if model_name is not None:
        # Load pre-trained tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        # Use a small, fast tokenizer (GPT-2 as base)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


def load_text_data(
    source: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    max_samples: Optional[int] = None,
    split: str = "train",
) -> Dataset:
    """
    Load text data from various sources.
    
    Supported sources:
    - File path (txt, json, jsonl)
    - HuggingFace dataset name
    - Directory of text files
    
    Args:
        source: Data source (path or dataset name)
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        max_samples: Maximum number of samples to load
        split: Dataset split to use
    
    Returns:
        Dataset instance
    """
    texts = []
    
    source_path = Path(source) if not '/' in source or Path(source).exists() else None
    
    if source_path and source_path.exists():
        if source_path.is_file():
            # Single file
            if source_path.suffix == '.txt':
                texts = [source_path.read_text(encoding='utf-8')]
            elif source_path.suffix in ['.json', '.jsonl']:
                with open(source_path, 'r', encoding='utf-8') as f:
                    if source_path.suffix == '.jsonl':
                        data = [json.loads(line) for line in f]
                    else:
                        data = json.load(f)
                
                # Extract text field
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            texts.append(item)
                        elif isinstance(item, dict):
                            texts.append(item.get('text', item.get('content', str(item))))
        elif source_path.is_dir():
            # Directory of files
            for file_path in source_path.glob('*.txt'):
                texts.append(file_path.read_text(encoding='utf-8'))
    
    elif DATASETS_AVAILABLE:
        # Try loading from HuggingFace
        try:
            dataset = load_dataset(source, split=split, streaming=False)
            
            # Find text column
            text_columns = ['text', 'content', 'document', 'sentence']
            text_col = None
            for col in text_columns:
                if col in dataset.column_names:
                    text_col = col
                    break
            
            if text_col is None:
                text_col = dataset.column_names[0]
            
            texts = dataset[text_col]
            
        except Exception as e:
            print(f"Failed to load dataset '{source}': {e}")
            raise
    
    else:
        raise ValueError(f"Cannot load data from '{source}'. Install 'datasets' for HuggingFace datasets.")
    
    # Limit samples
    if max_samples is not None and len(texts) > max_samples:
        texts = texts[:max_samples]
    
    print(f"Loaded {len(texts)} text samples")
    
    return TextDataset(texts, tokenizer, max_length=max_length)


def create_dataloaders(
    train_source: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    eval_source: Optional[str] = None,
    num_workers: int = 0,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = 1000,
) -> tuple:
    """
    Create training and evaluation dataloaders.
    
    Args:
        train_source: Training data source
        tokenizer: Tokenizer to use
        batch_size: Batch size
        max_length: Maximum sequence length
        eval_source: Evaluation data source (optional)
        num_workers: Number of data loading workers
        max_train_samples: Maximum training samples
        max_eval_samples: Maximum evaluation samples
    
    Returns:
        (train_dataloader, eval_dataloader)
    """
    # Create training dataset
    train_dataset = load_text_data(
        train_source,
        tokenizer,
        max_length=max_length,
        max_samples=max_train_samples,
        split="train",
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # Create evaluation dataset
    eval_dataloader = None
    if eval_source is not None:
        eval_dataset = load_text_data(
            eval_source,
            tokenizer,
            max_length=max_length,
            max_samples=max_eval_samples,
            split="validation",
        )
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return train_dataloader, eval_dataloader


# Synthetic data generation for testing
def generate_synthetic_math_data(
    num_samples: int = 1000,
    max_number: int = 100,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic math problems for reasoning training.
    
    Creates simple arithmetic problems with step-by-step solutions.
    """
    data = []
    operations = [
        ('addition', '+', lambda a, b: a + b),
        ('subtraction', '-', lambda a, b: a - b),
        ('multiplication', '*', lambda a, b: a * b),
    ]
    
    for _ in range(num_samples):
        a = random.randint(1, max_number)
        b = random.randint(1, max_number)
        op_name, op_symbol, op_func = random.choice(operations)
        
        result = op_func(a, b)
        
        reasoning = [
            f"We need to perform {op_name}",
            f"The first number is {a}",
            f"The second number is {b}",
            f"Computing {a} {op_symbol} {b}",
            f"The result is {result}",
        ]
        
        data.append({
            'question': f"What is {a} {op_symbol} {b}?",
            'reasoning': reasoning,
            'answer': str(result),
        })
    
    return data


if __name__ == "__main__":
    # Test data loading
    print("Testing data utilities...")
    
    # Create tokenizer
    tokenizer = create_tokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Generate synthetic data
    math_data = generate_synthetic_math_data(100)
    print(f"Generated {len(math_data)} synthetic math problems")
    
    # Create reasoning dataset
    dataset = ReasoningDataset(math_data, tokenizer, max_length=256)
    print(f"Dataset size: {len(dataset)}")
    
    # Test sample
    sample = dataset[0]
    print(f"\nSample:")
    print(f"  Input shape: {sample['input_ids'].shape}")
    print(f"  Decoded: {tokenizer.decode(sample['input_ids'][:100])}...")
    
    # Test dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    print(f"\nBatch:")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Labels shape: {batch['labels'].shape}")
