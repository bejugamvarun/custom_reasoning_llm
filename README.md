# Reasoning LLM with Mixture of Experts (MoE) and Chain of Thought (CoT)

A small, educational language model implementation designed to run on consumer GPUs (RTX 4070 8GB VRAM).

## 🎯 Purpose

This project is designed to help you understand the internal architecture of modern LLMs:

- **Transformer Architecture**: Multi-head attention, RoPE positional encoding
- **Mixture of Experts (MoE)**: Sparse expert routing, load balancing
- **Chain of Thought (CoT)**: Iterative reasoning module
- **Memory Efficient Training**: Mixed precision, gradient accumulation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│              Token Embeddings               │
│           + RoPE Positional Encoding        │
└──────────────────┬──────────────────────────┘
                   │
     ┌─────────────▼─────────────┐
     │    Transformer Block      │ × 6 layers
     │  ┌─────────────────────┐  │
     │  │  Multi-Head Attn    │  │
     │  │  (8 heads, RoPE)    │  │
     │  └─────────────────────┘  │
     │  ┌─────────────────────┐  │
     │  │   FFN or MoE        │  │
     │  │  (4 experts, top-2) │  │
     │  └─────────────────────┘  │
     └─────────────┬─────────────┘
                   │
     ┌─────────────▼─────────────┐
     │    Chain of Thought       │
     │   (Adaptive Reasoning)    │
     └─────────────┬─────────────┘
                   │
     ┌─────────────▼─────────────┐
     │       RMS Norm            │
     └─────────────┬─────────────┘
                   │
     ┌─────────────▼─────────────┐
     │      LM Head (tied)       │
     └───────────────────────────┘
```

## 📊 Model Configurations

| Config | Params | Hidden | Layers | Heads | MoE Experts | VRAM Est. |
|--------|--------|--------|--------|-------|-------------|-----------|
| tiny   | ~15M   | 256    | 4      | 4     | 2           | ~1 GB     |
| small  | ~50M   | 512    | 6      | 8     | 4           | ~2 GB     |
| medium | ~120M  | 768    | 8      | 12    | 8           | ~5 GB     |

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Test the Model

```bash
python main.py --test
```

### 3. Train on Synthetic Data

```bash
python main.py --train --synthetic --max-steps 1000
```

### 4. Train on Real Data

```bash
python main.py --train --dataset wikitext --subset wikitext-2-raw-v1 --max-steps 5000
```

### 5. Generate Text

```bash
python main.py --generate --prompt "The meaning of life is"
```

## 📁 Project Structure

```
custom_llm/
├── model/
│   ├── __init__.py
│   ├── config.py       # Model configuration
│   ├── attention.py    # Multi-head attention with RoPE
│   ├── moe.py          # Mixture of Experts implementation
│   ├── cot.py          # Chain of Thought module
│   └── transformer.py  # Full transformer architecture
├── training/
│   ├── __init__.py
│   ├── trainer.py      # Training loop with memory optimization
│   └── data.py         # Data loading utilities
├── main.py             # Entry point
├── pyproject.toml      # Project dependencies
└── README.md           # This file
```

## 🔧 Key Components Explained

### Rotary Positional Encoding (RoPE)

RoPE encodes position by rotating query and key vectors:
```python
# Rotation formula
x_rotated = x * cos(θ) + rotate_half(x) * sin(θ)
```

Benefits:
- Encodes relative position through dot product
- Extrapolates to longer sequences
- No learned position embeddings needed

### Mixture of Experts (MoE)

MoE replaces the standard FFN with multiple expert networks:
```python
# Router selects top-k experts per token
router_weights, selected_experts = router(hidden_states)

# Only selected experts are computed
output = sum(weight[i] * expert[i](input) for i in selected_experts)
```

Benefits:
- Increased model capacity
- Constant compute cost (sparse activation)
- Experts can specialize

### Chain of Thought (CoT)

CoT adds iterative reasoning capability:
```python
# Multiple reasoning steps with state evolution
for step in range(num_steps):
    reasoning_state = reasoning_cell(input, reasoning_state)

# Gated output blending
output = hidden_states + gate * reasoning_output
```

Benefits:
- Improved reasoning on complex tasks
- Interpretable reasoning traces
- Adaptive reasoning depth

## 💾 Memory Optimization for RTX 4070 (8GB)

The implementation includes several memory optimizations:

1. **Mixed Precision (FP16/BF16)**: Halves memory for activations
2. **Gradient Accumulation**: Effective large batches with small micro-batches
3. **Flash Attention**: O(n) memory instead of O(n²)
4. **Gradient Checkpointing**: Trade compute for memory (optional)

### Recommended Settings for 8GB VRAM

```python
TrainingConfig(
    micro_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch = 32
    use_amp=True,                    # Mixed precision
    amp_dtype="float16",
)
```

## 📈 Training Tips

1. **Start Small**: Test with `tiny` config first
2. **Monitor Memory**: Use `nvidia-smi` during training
3. **Learning Rate**: 3e-4 works well for small models
4. **Warmup**: Use 5-10% of training steps
5. **Gradient Clipping**: Keep at 1.0 for stability

## 🧪 Understanding the Code

Each module includes detailed docstrings explaining:
- Mathematical foundations
- Implementation choices
- Memory considerations

Start with `model/config.py` to understand the hyperparameters, then explore the modules in order:
1. `attention.py` - Attention mechanism
2. `moe.py` - Mixture of Experts
3. `cot.py` - Chain of Thought
4. `transformer.py` - Full model

## 📚 Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [RoFormer](https://arxiv.org/abs/2104.09864) - Rotary Position Embedding
- [Switch Transformer](https://arxiv.org/abs/2101.03961) - Mixture of Experts
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) - CoT reasoning

## 🤝 Contributing

This is an educational project. Feel free to:
- Add more detailed comments
- Implement additional features
- Improve memory efficiency
- Add visualization tools

## 📄 License

MIT License - Feel free to use for learning and experimentation!
