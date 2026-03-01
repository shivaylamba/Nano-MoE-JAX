#!/usr/bin/env python3
"""Train a character-level NanoMoE on Tiny Shakespeare.

Usage:
    python examples/train_shakespeare.py
"""

import jax
import jax.numpy as jnp

from nano_moe.config import NanoMoEConfig
from nano_moe.model import NanoMoE
from nano_moe.train import create_train_state, train
from nano_moe.utils import count_params, load_text_data


def main():
    # ---- Configuration ----
    # Override defaults for a quick demo — feel free to increase for better results
    config = NanoMoEConfig(
        n_layers=4,
        n_heads=4,
        d_model=128,
        d_ff=512,
        n_experts=4,
        top_k=2,
        block_size=128,
        # NanoMoE++ routing settings
        capacity_factor=1.25,
        router_jitter_noise=0.1,
        z_loss_coeff=1e-3,
        dropout_rate=0.1,
        aux_loss_coeff=0.01,
        learning_rate=3e-4,
        weight_decay=0.1,
        batch_size=32,
        max_iters=5000,
        eval_interval=250,
        eval_iters=50,
    )

    # ---- Data ----
    train_data, val_data, vocab_size, encode, decode = load_text_data()

    # Update config with actual vocab size
    config = NanoMoEConfig(
        vocab_size=vocab_size,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_model=config.d_model,
        d_ff=config.d_ff,
        n_experts=config.n_experts,
        top_k=config.top_k,
        block_size=config.block_size,
        capacity_factor=config.capacity_factor,
        router_jitter_noise=config.router_jitter_noise,
        z_loss_coeff=config.z_loss_coeff,
        dropout_rate=config.dropout_rate,
        aux_loss_coeff=config.aux_loss_coeff,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        max_iters=config.max_iters,
        eval_interval=config.eval_interval,
        eval_iters=config.eval_iters,
    )

    print("\n" + "=" * 60)
    print("  NanoMoE++ — Mixture-of-Experts Language Model (JAX)")
    print("=" * 60)
    print(f"  Layers: {config.n_layers}  |  Heads: {config.n_heads}  |  "
          f"d_model: {config.d_model}")
    print(f"  Experts: {config.n_experts}  |  Top-K: {config.top_k}  |  "
          f"d_ff: {config.d_ff}")
    print(f"  Capacity factor: {config.capacity_factor}  |  "
          f"Jitter: {config.router_jitter_noise}  |  Z-loss coeff: {config.z_loss_coeff}")
    print(f"  Block size: {config.block_size}  |  Vocab: {config.vocab_size}")

    # ---- Init ----
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    n_params = count_params(state.params)
    print(f"  Total parameters: {n_params:,}")
    print("=" * 60 + "\n")

    # ---- Train ----
    state = train(state, config, train_data, val_data, rng)

    # ---- Generate ----
    print("\n" + "=" * 60)
    print("  Generating sample text …")
    print("=" * 60 + "\n")

    model = NanoMoE(config=config)
    prompt_text = "\n"  # start with a newline
    prompt = jnp.array([encode(prompt_text)], dtype=jnp.int32)

    rng, gen_rng = jax.random.split(rng)
    generated = model.generate(
        state.params, gen_rng, prompt,
        max_new_tokens=500, temperature=0.8, top_k_sample=40,
    )

    output = decode(generated[0].tolist())
    print(output)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
