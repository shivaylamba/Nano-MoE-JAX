# NanoMoE++ — Smarter Sparse MoE in JAX/Flax

A lightweight, educational **Mixture-of-Experts (MoE)** GPT-style language model built from scratch in **JAX / Flax** — upgraded beyond classic NanoMoE with capacity-aware routing and expert parallelism-friendly design.

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT), NanoMoE++ replaces the standard FFN in each transformer block with a Switch/Mixtral-style sparse MoE, but improves quality and stability by adding:

- **Capacity-limited routing** — prevents expert overload and training collapse
- **Top-k gating + optional jitter noise** — better exploration, less expert collapse
- **Aux load-balancing + z-loss** — more uniform utilisation, steadier routing logits
- **Token dispatch/collect primitives** — cleaner, faster JIT-friendly sparse execution

Only top-k experts run per token, so you get more capacity per FLOP while keeping the codebase compact and readable.

## Architecture

```
Input Tokens
    ↓
Token Embedding + Positional Embedding
    ↓
┌──────────────────────────────────────────────────┐
│           Transformer Block × N                  │
│                                                  │
│  LayerNorm → Causal Multi-Head Attention         │
│      ↓ + Residual                                │
│  LayerNorm → MoE++ Layer                         │
│      ↓ + Residual                                │
│                                                  │
│  ┌─── MoE++ Layer ────────────────────────────┐  │
│  │ Router (Top-K + Jitter Noise + Z-Loss)     │  │
│  │   ├─ dispatch tokens → Expert 1 (FFN)      │  │
│  │   ├─ dispatch tokens → Expert 2 (FFN)      │  │
│  │   ├─ ...                                   │  │
│  │   └─ dispatch tokens → Expert N (FFN)      │  │
│  │ Capacity check (drop overflow tokens)      │  │
│  │ → collect + weighted sum via gates         │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
    ↓
LayerNorm → Linear Head → Logits
```
<img width="400" height="800" alt="image" src="https://github.com/user-attachments/assets/61c93464-e2c3-4fac-a089-5ce327fdec21" />

### Key Features

| Feature | NanoMoE (original) | NanoMoE++ |
|---|---|---|
| Sparse Top-K routing | ✓ | ✓ |
| Load-balancing aux loss | ✓ | ✓ |
| **Capacity-limited dispatch** | ✗ | ✓ |
| **Token dispatch / collect** | ✗ | ✓ |
| **Z-loss (routing stability)** | ✗ | ✓ |
| **Jitter noise on logits** | ✗ | ✓ |
| Pure JAX/Flax (no custom CUDA) | ✓ | ✓ |
| Autoregressive generation | ✓ | ✓ |

## Quick Start

### Install

```bash
git clone https://github.com/shivaylamba/Nano-MoE-JAX.git
cd Nano-MoE-JAX
pip install -r requirements.txt
```

> **Note:** For GPU support, install the appropriate `jaxlib` CUDA wheel — see [JAX installation](https://github.com/google/jax#installation).

### Train on Tiny Shakespeare

```bash
python examples/train_shakespeare.py
```

This downloads Tiny Shakespeare (~1 MB), trains a character-level NanoMoE++, and generates sample text.

### Run Tests

```bash
python -m pytest tests/ -v
```

### Run Architecture Comparison Benchmark

```bash
python tests/test_benchmark.py
```

## Project Structure

```
Nano-MoE-JAX/
├── nano_moe/
│   ├── __init__.py        # Public API
│   ├── config.py          # Hyperparameter dataclass (incl. capacity_factor, z_loss_coeff)
│   ├── layers.py          # ExpertFFN, Router (z-loss + jitter), MoELayer (dispatch/collect)
│   ├── model.py           # NanoMoE model + generate()
│   ├── train.py           # Training loop, JIT-compiled steps
│   └── utils.py           # Param counting, batching, data loading
├── examples/
│   └── train_shakespeare.py
├── tests/
│   ├── test_layers.py     # Unit tests for all layers (incl. NanoMoE++ features)
│   ├── test_model.py      # End-to-end model tests
│   └── test_benchmark.py  # Latency & convergence comparison: baseline vs NanoMoE++
├── requirements.txt
└── README.md
```

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `d_model` | 128 | Hidden dimension |
| `n_layers` | 4 | Transformer blocks |
| `n_heads` | 4 | Attention heads |
| `d_ff` | 512 | Expert FFN inner dim |
| `n_experts` | 4 | Experts per MoE layer |
| `top_k` | 2 | Active experts per token |
| `block_size` | 128 | Max context length |
| `capacity_factor` | 1.25 | Expert buffer size multiplier (≥ top_k → no dropping) |
| `router_jitter_noise` | 0.0 | Logit noise half-width during training |
| `aux_loss_coeff` | 0.01 | Load-balancing loss weight |
| `z_loss_coeff` | 0.001 | Routing logit stability loss weight |

## How NanoMoE++ Works

### 1. Router with Z-Loss and Jitter Noise

```python
# Raw routing logits
logits = Dense(n_experts)(x)                       # (B, T, E)

# Optional training-time jitter for exploration
if not deterministic and jitter_noise > 0:
    logits += Uniform(-jitter_noise, +jitter_noise)

# Z-loss: penalises large logit magnitudes (ST-MoE, Zoph et al. 2022)
z_loss = mean(logsumexp(logits, axis=-1) ** 2)

# Top-k selection + softmax gate normalisation
top_k_logits, indices = top_k(logits, k)
gates = softmax(top_k_logits)                      # (B, T, K)
```

### 2. Capacity-Aware Token Dispatch / Collect

```python
# Each expert's capacity buffer: ceil(capacity_factor * N / n_experts)
capacity = ceil(capacity_factor * N / n_experts)

# Assign tokens to expert slots via cumulative count
slot_indices = cumsum(one_hot(indices)) - 1        # (N, K)
capacity_mask = slot_indices < capacity            # drop overflow

# dispatch[e, c, n] = 1 iff token n fills slot c of expert e
dispatch = einsum('nke,nkc->ecn', expert_mask, slot_one_hot)

# Gather → run experts on (C, D) batches → scatter back
expert_input  = einsum('ecn,nd->ecd', dispatch, tokens)
expert_output = stack([expert_i(expert_input[i]) ...])    # (E, C, D)
output = einsum('ne,end->nd', per_expert_gate,
                einsum('ecn,ecd->end', dispatch, expert_output))
```

### 3. Combined Auxiliary Loss

```python
combined_aux = aux_loss + z_loss_coeff * z_loss
total_loss   = ce_loss + aux_loss_coeff * combined_aux
```

## Architecture Comparison

Benchmarked on identical model configs (d_model=32, 2 layers, 4 experts, top-k=2). Lower latency and CE are better:

| Architecture | Params | Latency (ms) ↓ | Init CE | Final CE (50 steps) ↓ | CE drop ↑ | Aux loss ↓ |
|---|---|---|---|---|---|---|
| NanoMoE (baseline) | 47,232 | 0.888 | 4.1525 | 4.1023 | 0.0502 | 6.1599 |
| NanoMoE++ | 47,232 | **0.298** | 4.1918 | 4.1312 | **0.0605** | **4.1957** |

**Key observations:**

- **Same parameter count** — NanoMoE++ introduces no extra parameters; routing improvements are purely algorithmic.
- **3× faster forward pass** — Capacity-limited dispatch means each expert runs on a smaller, bounded batch (`C` tokens) instead of all `B×T` tokens.
- **Larger CE drop** — NanoMoE++ drops 0.0605 nats vs 0.0502 for baseline over 50 steps (20% faster convergence rate).
- **Lower aux loss** — Capacity masking and z-loss together reduce routing pressure, indicating more balanced and stable expert utilisation.

To reproduce:
```bash
python tests/test_benchmark.py
```

## License

MIT
