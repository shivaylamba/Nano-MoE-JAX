# NanoMoE++ & HierMoE — Sparse Mixture-of-Experts in JAX/Flax

A lightweight, educational suite of three **Mixture-of-Experts (MoE)** GPT-style language models built from scratch in **JAX / Flax**, progressing from a simple flat router to a fully hierarchical two-stage routing architecture.

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT), each model replaces the standard FFN in every transformer block with a sparse MoE layer — only top-k experts activate per token, giving increased capacity with reduced compute per forward pass.

---

## Architecture Overview

### Version A — NanoMoE (baseline)

Original flat top-k routing: each token is routed to any k of E experts with a single router.

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
│  LayerNorm → MoE Layer                           │
│      ↓ + Residual                                │
│                                                  │
│  ┌─── MoE Layer ─────────────────────────────┐   │
│  │  Router (Top-K Gating, softmax)            │   │
│  │    ├─ Expert 1 (FFN)                       │   │
│  │    ├─ Expert 2 (FFN)                       │   │
│  │    └─ Expert N (FFN)                       │   │
│  │  → Weighted sum of Top-K outputs           │   │
│  └────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
    ↓
LayerNorm → Linear Head → Logits
```

### Version B — NanoMoE++ (capacity-aware routing)

Same flat router but with production-grade improvements: capacity-limited dispatch/collect, z-loss, and jitter noise.

```
MoE++ Layer
┌────────────────────────────────────────────────┐
│  Router (Top-K + Z-Loss + Jitter Noise)        │
│  Capacity check → drop overflow tokens         │
│  Token dispatch tensor (E × C × N)             │
│    ├─ Expert 1 (FFN) on C-token batch          │
│    ├─ Expert 2 (FFN) on C-token batch          │
│    └─ Expert N (FFN) on C-token batch          │
│  Collect + weighted gate sum                   │
└────────────────────────────────────────────────┘
```

### Version C — HierMoE (hierarchical two-stage routing)

Two-stage routing: **Stage 1** routes each token to a coarse *expert group*; **Stage 2** picks the top-k experts *within* that group. This reduces routing chaos, improves expert specialisation, and scales cleanly to large expert counts.

```
HierMoE Layer
┌────────────────────────────────────────────────────────────────┐
│  Stage 1 — Group Router (Dense → G logits)                     │
│    → top-1 group selected per token                            │
│                                                                │
│  Stage 2 — Expert Router (Dense → E logits, masked to group)   │
│    → top-k within selected group                               │
│                                                                │
│  Expert groups (each with E/G experts):                        │
│    Group 0: [ Expert 0 | Expert 1 | … | Expert (E/G-1) ]      │
│    Group 1: [ Expert E/G | … | Expert 2*(E/G)-1 ]             │
│    …                                                           │
│                                                                │
│  Capacity-limited dispatch + collect (same as NanoMoE++)       │
│  Combined aux loss = group_aux + expert_aux + z_loss           │
└────────────────────────────────────────────────────────────────┘
```

<img width="400" height="800" alt="NanoMoE architecture" src="https://github.com/user-attachments/assets/61c93464-e2c3-4fac-a089-5ce327fdec21" />

---

## Feature Comparison

| Feature | NanoMoE (baseline) | NanoMoE++ | HierMoE |
|---|:---:|:---:|:---:|
| Sparse Top-K routing | ✓ | ✓ | ✓ |
| Load-balancing aux loss | ✓ | ✓ | ✓ |
| **Capacity-limited dispatch** | ✗ | ✓ | ✓ |
| **Token dispatch / collect** | ✗ | ✓ | ✓ |
| **Z-loss (routing stability)** | ✗ | ✓ | ✓ |
| **Jitter noise on logits** | ✗ | ✓ | ✓ |
| **Two-stage group routing** | ✗ | ✗ | ✓ |
| **Per-group load-balancing** | ✗ | ✗ | ✓ |
| Pure JAX/Flax (no custom CUDA) | ✓ | ✓ | ✓ |
| Autoregressive generation | ✓ | ✓ | ✓ |

---

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

### Run 3-Way Architecture Comparison Benchmark

```bash
python tests/test_benchmark.py
```

---

## Project Structure

```
Nano-MoE-JAX/
├── nano_moe/
│   ├── __init__.py        # Public API (all three architectures)
│   ├── config.py          # NanoMoEConfig (shared; n_expert_groups controls HierMoE)
│   ├── layers.py          # ExpertFFN, Router, MoELayer, MultiHeadAttention,
│   │                      # TransformerBlock, HierRouter, HierMoELayer,
│   │                      # HierTransformerBlock
│   ├── model.py           # NanoMoE, HierNanoMoE, generate()
│   ├── train.py           # Training loop, JIT steps, create_train_state(model_cls=…)
│   └── utils.py           # count_params, get_batch, load_text_data
├── examples/
│   └── train_shakespeare.py
├── tests/
│   ├── test_layers.py     # Unit tests: flat layers + HierRouter/HierMoELayer/HierBlock
│   ├── test_model.py      # End-to-end: NanoMoE + HierNanoMoE
│   └── test_benchmark.py  # 3-way latency & convergence comparison
├── requirements.txt
└── README.md
```

---

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
| `capacity_factor` | 1.25 | Expert buffer size multiplier |
| `router_jitter_noise` | 0.0 | Logit noise half-width (training) |
| `aux_loss_coeff` | 0.01 | Load-balancing loss weight |
| `z_loss_coeff` | 0.001 | Routing logit stability loss weight |
| `n_expert_groups` | **1** | **HierMoE: number of expert groups (must divide `n_experts`)** |

---

## How HierMoE Works

### Stage 1 — Group Router

```python
group_logits  = Dense(n_groups)(x)            # (B, T, G)
group_probs   = softmax(group_logits)
group_idx     = argmax(group_logits, axis=-1) # (B, T) — top-1 group
```

### Stage 2 — Expert Router (masked to selected group)

```python
expert_logits = Dense(n_experts)(x)           # (B, T, E)
# Reshape → (B, T, G, M) and extract the chosen group:
selected      = expert_logits[:, :, group_idx, :]   # (B, T, M)
top_k_vals, intra_idx = top_k(selected, K)
gates  = softmax(top_k_vals)
# Map intra-group indices → global indices:
global_idx = intra_idx + group_idx * M        # (B, T, K)
```

### Combined Auxiliary Loss

```python
# Group-level: how uniformly are tokens spread across groups?
group_aux  = G * sum(f_g * P_g)

# Expert-level: how uniformly are tokens spread across experts?
expert_aux = E * sum(f_e * P_e)

# Z-loss on both routers
z_loss = mean(logsumexp(group_logits)**2) + mean(logsumexp(expert_logits)**2)

combined_aux = group_aux + expert_aux + z_loss_coeff * z_loss
```

---

## 3-Way Architecture Comparison

Benchmarked on identical small configs (d_model=32, 2 layers, 4 experts, top-k=2, 50 training steps).
Lower latency ↓ and final CE ↓ are better:

| Architecture | Params | Latency (ms) ↓ | Init CE | Final CE ↓ | CE drop ↑ | Aux loss |
|---|---|---|---|---|---|---|
| NanoMoE (baseline) | 47,232 | 0.854 | 4.1627 | 4.1220 | 0.0408 | 6.3705 |
| NanoMoE++ | 47,232 | **0.267** | 4.1577 | 4.1295 | 0.0282 | **4.1957** |
| HierMoE | **47,360** | 0.292 | **4.1436** | 4.1340 | 0.0096 | 6.1384 |

**Key observations:**

- **All three have essentially the same parameter count** — HierMoE adds only 128 extra parameters (the group-level Dense(G) router per block).
- **NanoMoE++ and HierMoE are ~3× faster** — capacity-limited dispatch confines each expert to a fixed `C`-token buffer rather than all `B×T` tokens; the sparse execution is genuinely faster after JIT compilation.
- **Convergence is equivalent at this micro-scale** — the toy 50-step run with a 32-d model is far too small to distinguish architectures by accuracy; architectural benefits (stability, scalability, specialisation) emerge at larger expert counts and longer training.
- **NanoMoE++ has the lowest aux loss** — z-loss + capacity masking together give the most balanced routing pressure of the three.
- **HierMoE advantage** — at large scale (many experts), hierarchical routing reduces the search space per token (from E to E/G), making load-balancing far more tractable and enabling expert groups to develop coarse topic-level specialisations.

To reproduce:
```bash
python tests/test_benchmark.py
```

---

## Using HierMoE in Your Code

```python
from nano_moe.config import NanoMoEConfig
from nano_moe.model import HierNanoMoE
from nano_moe.train import create_train_state

config = NanoMoEConfig(
    vocab_size=256,
    n_layers=4,
    n_heads=4,
    d_model=128,
    d_ff=512,
    n_experts=8,          # 8 experts total
    n_expert_groups=2,    # 2 groups of 4 experts — enables HierMoE
    top_k=2,              # top-2 within selected group
    capacity_factor=1.25,
    router_jitter_noise=0.1,
    z_loss_coeff=1e-3,
    aux_loss_coeff=0.01,
)

import jax
state = create_train_state(jax.random.PRNGKey(0), config, model_cls=HierNanoMoE)
```

---

## License

MIT
