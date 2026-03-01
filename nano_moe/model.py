"""NanoMoE — Full model: embedding → transformer blocks → language model head."""

from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from nano_moe.config import NanoMoEConfig
from nano_moe.layers import TransformerBlock, HierTransformerBlock


class NanoMoE(nn.Module):
    """Nano Mixture-of-Experts language model.

    Architecture:
        Token embedding + learned positional embedding
        → N × TransformerBlock (each with MoE layer)
        → LayerNorm → Linear head → logits
    """

    config: NanoMoEConfig

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            x: Integer token indices of shape (batch, seq_len).
            deterministic: If True, disable dropout.

        Returns:
            logits:   (batch, seq_len, vocab_size)
            aux_loss: Scalar — sum of load-balancing losses from all blocks.
        """
        cfg = self.config
        B, T = x.shape

        # ---------- Embeddings ----------
        tok_emb = nn.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )(x)  # (B, T, D)

        pos_emb = self.param(
            "pos_emb",
            nn.initializers.normal(stddev=0.02),
            (1, cfg.block_size, cfg.d_model),
        )  # (1, block_size, D)
        x = tok_emb + pos_emb[:, :T, :]  # (B, T, D)

        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)

        # ---------- Transformer blocks ----------
        total_aux_loss = jnp.float32(0.0)

        for i in range(cfg.n_layers):
            x, aux_loss = TransformerBlock(config=cfg, name=f"block_{i}")(
                x, deterministic=deterministic
            )
            total_aux_loss = total_aux_loss + aux_loss

        # ---------- Head ----------
        x = nn.LayerNorm()(x)
        logits = nn.Dense(
            cfg.vocab_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(x)  # (B, T, V)

        return logits, total_aux_loss

    def generate(
        self,
        params,
        rng: jax.Array,
        prompt: jnp.ndarray,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k_sample: int = 40,
    ) -> jnp.ndarray:
        """Autoregressive token generation.

        Args:
            params: Model parameters.
            rng: PRNG key.
            prompt: (1, T) integer token indices.
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature.
            top_k_sample: Top-k filtering before sampling.

        Returns:
            tokens: (1, T + max_new_tokens) generated sequence.
        """
        cfg = self.config
        tokens = prompt  # (1, T)

        for _ in range(max_new_tokens):
            # Crop to block_size if needed
            x_cond = tokens[:, -cfg.block_size:]

            # Forward pass (deterministic = True for generation)
            logits, _ = self.apply({"params": params}, x_cond, deterministic=True)

            # Take logits at the last position and apply temperature
            next_logits = logits[:, -1, :] / temperature  # (1, V)

            # Top-k filtering
            if top_k_sample > 0:
                top_vals, _ = jax.lax.top_k(next_logits, top_k_sample)
                threshold = top_vals[:, -1:]
                next_logits = jnp.where(next_logits < threshold, -1e9, next_logits)

            # Sample
            rng, sample_rng = jax.random.split(rng)
            next_token = jax.random.categorical(sample_rng, next_logits, axis=-1)  # (1,)
            next_token = next_token[:, None]  # (1, 1)
            tokens = jnp.concatenate([tokens, next_token], axis=1)

        return tokens


class HierNanoMoE(nn.Module):
    """Hierarchical Mixture-of-Experts language model.

    Identical to :class:`NanoMoE` except each transformer block uses
    :class:`~nano_moe.layers.HierTransformerBlock` with two-stage
    (group → expert) routing provided by
    :class:`~nano_moe.layers.HierRouter`.

    Requires ``config.n_expert_groups > 1`` and
    ``config.n_experts % config.n_expert_groups == 0``.
    """

    config: NanoMoEConfig

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cfg = self.config
        B, T = x.shape

        tok_emb = nn.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )(x)

        pos_emb = self.param(
            "pos_emb",
            nn.initializers.normal(stddev=0.02),
            (1, cfg.block_size, cfg.d_model),
        )
        x = tok_emb + pos_emb[:, :T, :]
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)

        total_aux_loss = jnp.float32(0.0)
        for i in range(cfg.n_layers):
            x, aux_loss = HierTransformerBlock(config=cfg, name=f"block_{i}")(
                x, deterministic=deterministic
            )
            total_aux_loss = total_aux_loss + aux_loss

        x = nn.LayerNorm()(x)
        logits = nn.Dense(
            cfg.vocab_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(x)

        return logits, total_aux_loss

    def generate(
        self,
        params,
        rng: jax.Array,
        prompt: jnp.ndarray,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k_sample: int = 40,
    ) -> jnp.ndarray:
        """Autoregressive token generation (identical API to :class:`NanoMoE`)."""
        cfg = self.config
        tokens = prompt

        for _ in range(max_new_tokens):
            x_cond = tokens[:, -cfg.block_size:]
            logits, _ = self.apply({"params": params}, x_cond, deterministic=True)
            next_logits = logits[:, -1, :] / temperature

            if top_k_sample > 0:
                top_vals, _ = jax.lax.top_k(next_logits, top_k_sample)
                threshold = top_vals[:, -1:]
                next_logits = jnp.where(next_logits < threshold, -1e9, next_logits)

            rng, sample_rng = jax.random.split(rng)
            next_token = jax.random.categorical(sample_rng, next_logits, axis=-1)
            next_token = next_token[:, None]
            tokens = jnp.concatenate([tokens, next_token], axis=1)

        return tokens
