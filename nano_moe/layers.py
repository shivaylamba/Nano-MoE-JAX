"""Core layers: Expert FFN, Router, MoE, Multi-Head Attention, Transformer Block."""

import math
from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from nano_moe.config import NanoMoEConfig


# ---------------------------------------------------------------------------
# Expert Feed-Forward Network
# ---------------------------------------------------------------------------

class ExpertFFN(nn.Module):
    """Two-layer FFN with GELU activation (a single expert).

    Architecture: d_model → d_ff → d_model
    """

    d_ff: int
    d_model: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.d_ff, kernel_init=nn.initializers.he_normal())(x)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model, kernel_init=nn.initializers.he_normal())(x)
        return x


# ---------------------------------------------------------------------------
# Router / Gating Network
# ---------------------------------------------------------------------------

class Router(nn.Module):
    """Top-k gating network that routes tokens to experts.

    Improvements over vanilla top-k routing:

    * **Jitter noise** — small uniform noise added to logits during training
      encourages exploration and reduces expert collapse.
    * **Z-loss** — penalises large logit magnitudes (per ST-MoE) to keep
      routing distributions numerically stable.
    * **Aux load-balancing loss** — Switch Transformer-style loss that pushes
      the router towards uniform expert utilisation.
    """

    n_experts: int
    top_k: int
    jitter_noise: float = 0.0

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Route tokens to experts.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            deterministic: If False, apply jitter noise to logits.

        Returns:
            gates:     (batch, seq_len, top_k) — softmax weights for selected experts.
            indices:   (batch, seq_len, top_k) — indices of selected experts.
            aux_loss:  Scalar load-balancing loss.
            z_loss:    Scalar z-loss for routing logit stability.
        """
        # Compute raw logits → (batch, seq_len, n_experts)
        logits = nn.Dense(self.n_experts, use_bias=False,
                          kernel_init=nn.initializers.xavier_uniform())(x)

        # Optional jitter noise during training (improves expert exploration)
        if not deterministic and self.jitter_noise > 0.0:
            noise = jax.random.uniform(
                self.make_rng('dropout'),
                logits.shape,
                minval=-self.jitter_noise,
                maxval=self.jitter_noise,
            )
            logits = logits + noise

        # ---- Z-loss: E[log(sum(exp(logits)))^2] (ST-MoE, Zoph et al. 2022) ----
        # Penalises large logit magnitudes; keeps routing probabilities well-scaled.
        z_loss = jnp.mean(jax.nn.logsumexp(logits, axis=-1) ** 2)

        # Full softmax over experts for load-balance computation
        probs = jax.nn.softmax(logits, axis=-1)  # (B, T, E)

        # Top-k selection
        top_k_values, top_k_indices = jax.lax.top_k(logits, self.top_k)  # (B, T, K)

        # Normalised gates only over the selected experts
        gates = jax.nn.softmax(top_k_values, axis=-1)  # (B, T, K)

        # ---- Auxiliary load-balancing loss (Switch Transformer style) ----
        # f_i = fraction of tokens routed to expert i  (top-1 dispatch fraction)
        # P_i = mean routing probability for expert i
        # aux_loss = n_experts * sum_i(f_i * P_i)
        flat_top1 = top_k_indices.reshape(-1, self.top_k)[:, 0]     # (N,)
        flat_probs = probs.reshape(-1, self.n_experts)                # (N, E)
        dispatch_mask = jax.nn.one_hot(flat_top1, self.n_experts)    # (N, E)
        f = jnp.mean(dispatch_mask, axis=0)                          # (E,)
        P = jnp.mean(flat_probs, axis=0)                             # (E,)
        aux_loss = self.n_experts * jnp.sum(f * P)

        return gates, top_k_indices, aux_loss, z_loss


# ---------------------------------------------------------------------------
# Mixture-of-Experts Layer
# ---------------------------------------------------------------------------

class MoELayer(nn.Module):
    """Capacity-aware Mixture-of-Experts layer using token dispatch/collect.

    Key improvements over the vanilla MoE layer:

    * **Capacity-limited routing** — each expert processes at most
      ``ceil(capacity_factor * N / n_experts)`` tokens per forward pass,
      preventing any single expert from being overloaded.
    * **Token dispatch / collect** — tokens are gathered into per-expert
      batches of fixed size ``C`` (capacity), run through each expert
      independently, then scattered back; JIT-friendly with static shapes.
    * **Combined auxiliary losses** — aux load-balancing loss plus z-loss
      (weighted by ``z_loss_coeff``) are returned as a single scalar so the
      calling code needs no changes.
    """

    config: NanoMoEConfig

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model)
            deterministic: If True, disable dropout and jitter noise.

        Returns:
            output:       (batch, seq_len, d_model) — weighted expert outputs.
            combined_aux: Scalar — aux load-balancing loss + z_loss_coeff * z_loss.
        """
        cfg = self.config
        B, T, D = x.shape
        N = B * T
        E, K = cfg.n_experts, cfg.top_k

        # ---- Router ----
        gates, indices, aux_loss, z_loss = Router(
            n_experts=E, top_k=K, jitter_noise=cfg.router_jitter_noise,
        )(x, deterministic=deterministic)
        # gates: (B, T, K), indices: (B, T, K) → flatten to (N, K)
        gates_flat = gates.reshape(N, K)
        indices_flat = indices.reshape(N, K)

        # ---- Capacity per expert ----
        # capacity = ceil(capacity_factor * N / n_experts).
        # Assignments beyond this limit are zero-gated (tokens are dropped).
        capacity = max(1, math.ceil(cfg.capacity_factor * N / E))

        # ---- Build dispatch tensor (static shapes, JIT-friendly) ----
        # expert_mask[n, k, e] = 1  iff token n's k-th selection is expert e
        expert_mask = jax.nn.one_hot(indices_flat, E)  # (N, K, E)

        # Cumulative slot counter per expert — determines which slot each
        # (token, k) assignment occupies inside its expert's capacity buffer.
        flat_mask = expert_mask.reshape(N * K, E)            # (N*K, E)
        cumcounts = jnp.cumsum(flat_mask, axis=0)            # (N*K, E)
        # slot_indices[n, k] = 0-based slot index for assignment (n, k)
        slot_indices = jnp.sum(
            (cumcounts.reshape(N, K, E) - 1) * expert_mask, axis=-1
        ).astype(jnp.int32)  # (N, K)

        # Mask out assignments that exceed expert capacity
        capacity_mask = slot_indices < capacity              # (N, K)
        effective_gates = jnp.where(capacity_mask, gates_flat, 0.0)

        # Clamp slot indices so one_hot below never goes out of range
        safe_slots = jnp.clip(slot_indices, 0, capacity - 1)  # (N, K)

        # dispatch[e, c, n] = 1  iff token n fills capacity slot c of expert e
        slot_oh = jax.nn.one_hot(safe_slots, capacity)          # (N, K, C)
        dispatch = jnp.einsum(
            'nke,nkc->ecn',
            expert_mask * capacity_mask[..., None],
            slot_oh,
        )  # (E, C, N)

        # ---- Gather tokens into per-expert batches ----
        tokens = x.reshape(N, D)                              # (N, D)
        expert_input = jnp.einsum('ecn,nd->ecd', dispatch, tokens)  # (E, C, D)

        # ---- Run each expert on its (capacity-sized) token batch ----
        experts = [
            ExpertFFN(d_ff=cfg.d_ff, d_model=D, name=f"expert_{i}")
            for i in range(E)
        ]
        expert_outputs = jnp.stack(
            [experts[i](expert_input[i]) for i in range(E)], axis=0
        )  # (E, C, D)

        # ---- Collect expert outputs back to token positions ----
        # per_expert_output[e, n, :] = expert e's output for token n
        per_expert_output = jnp.einsum(
            'ecn,ecd->end', dispatch, expert_outputs
        )  # (E, N, D)

        # Weight by effective gates and sum across experts
        per_expert_gate = jnp.einsum(
            'nk,nke->ne', effective_gates, expert_mask
        )  # (N, E)
        output_flat = jnp.einsum(
            'ne,end->nd', per_expert_gate, per_expert_output
        )  # (N, D)

        output = output_flat.reshape(B, T, D)

        # Optional dropout on the combined output
        output = nn.Dropout(rate=cfg.dropout_rate)(output, deterministic=deterministic)

        # Combined auxiliary loss
        combined_aux = aux_loss + cfg.z_loss_coeff * z_loss

        return output, combined_aux


# ---------------------------------------------------------------------------
# Multi-Head Causal Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard multi-head causal self-attention with dropout."""

    config: NanoMoEConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        cfg = self.config
        B, T, D = x.shape
        head_dim = D // cfg.n_heads

        # QKV projection
        qkv = nn.Dense(3 * D, kernel_init=nn.initializers.xavier_uniform())(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each (B, T, D)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.reshape(B, T, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention with causal mask
        scale = jnp.sqrt(jnp.float32(head_dim))
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale  # (B, H, T, T)

        # Causal mask: upper-triangular → −∞
        causal_mask = jnp.triu(jnp.ones((T, T), dtype=jnp.bool_), k=1)
        attn_weights = jnp.where(causal_mask[None, None, :, :], -1e9, attn_weights)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = nn.Dropout(rate=cfg.dropout_rate)(attn_weights, deterministic=deterministic)

        # Weighted sum → (B, H, T, head_dim) → (B, T, D)
        attn_out = jnp.matmul(attn_weights, v)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, D)

        # Output projection
        out = nn.Dense(D, kernel_init=nn.initializers.xavier_uniform())(attn_out)
        out = nn.Dropout(rate=cfg.dropout_rate)(out, deterministic=deterministic)
        return out


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → Attention → residual, LN → MoE → residual."""

    config: NanoMoEConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Returns:
            output:   (B, T, D) — block output.
            aux_loss: Scalar MoE combined auxiliary loss (load-balance + z-loss).
        """
        # Self-attention sub-layer
        residual = x
        x = nn.LayerNorm()(x)
        x = MultiHeadAttention(config=self.config)(x, deterministic=deterministic)
        x = x + residual

        # MoE sub-layer
        residual = x
        x_norm = nn.LayerNorm()(x)
        moe_out, aux_loss = MoELayer(config=self.config)(x_norm, deterministic=deterministic)
        x = moe_out + residual

        return x, aux_loss


# ---------------------------------------------------------------------------
# Hierarchical Router (HierMoE)
# ---------------------------------------------------------------------------

class HierRouter(nn.Module):
    """Two-stage hierarchical routing network.

    Stage 1 — *coarse*: projects each token to ``n_expert_groups`` logits and
    selects the **top-1 group** via a group-level router.

    Stage 2 — *fine*: projects each token to ``n_experts`` logits, masks to
    the selected group, and picks the **top-k experts** within that group.

    This reduces routing chaos for large expert counts, encourages coarse
    topic-level specialisation at the group level, and fine-grained skill
    specialisation at the expert level.

    Args:
        n_expert_groups: Number of expert groups (G).  Must evenly divide
            ``n_experts``.
        n_experts: Total expert count (E).
        top_k: Experts activated per token from the selected group (K).
        jitter_noise: Logit jitter half-width (0 = disabled).
    """

    n_expert_groups: int
    n_experts: int
    top_k: int
    jitter_noise: float = 0.0

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Route tokens via two-stage hierarchy.

        Args:
            x: (batch, seq_len, d_model)
            deterministic: If False, apply jitter noise.

        Returns:
            gates:       (batch, seq_len, top_k) — normalised weights for
                         the selected global experts.
            indices:     (batch, seq_len, top_k) — global expert indices.
            aux_loss:    Scalar — group-level + expert-level load-balance loss.
            z_loss:      Scalar — z-loss on both group and expert logits.
        """
        G = self.n_expert_groups
        E = self.n_experts
        K = self.top_k
        M = E // G  # experts per group

        B, T, _ = x.shape

        # ---- Stage 1: group router (B, T, G) ----
        group_logits = nn.Dense(
            G, use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            name="group_router",
        )(x)

        # ---- Stage 2: expert router (B, T, E) ----
        expert_logits = nn.Dense(
            E, use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            name="expert_router",
        )(x)

        # Optional jitter noise during training
        if not deterministic and self.jitter_noise > 0.0:
            rng = self.make_rng("dropout")
            rng_g, rng_e = jax.random.split(rng)
            group_logits = group_logits + jax.random.uniform(
                rng_g, group_logits.shape,
                minval=-self.jitter_noise, maxval=self.jitter_noise,
            )
            expert_logits = expert_logits + jax.random.uniform(
                rng_e, expert_logits.shape,
                minval=-self.jitter_noise, maxval=self.jitter_noise,
            )

        # ---- Z-loss on both routers ----
        z_loss = (
            jnp.mean(jax.nn.logsumexp(group_logits, axis=-1) ** 2) +
            jnp.mean(jax.nn.logsumexp(expert_logits, axis=-1) ** 2)
        )

        # ---- Stage 1: select top-1 group per token ----
        group_probs = jax.nn.softmax(group_logits, axis=-1)        # (B, T, G)
        _, group_idx_raw = jax.lax.top_k(group_logits, 1)          # (B, T, 1)
        group_idx = group_idx_raw[..., 0]                           # (B, T)

        # ---- Stage 2: mask expert logits to the selected group ----
        # Reshape expert logits → (B, T, G, M)
        expert_logits_grouped = expert_logits.reshape(B, T, G, M)

        # For each token, extract logits of its chosen group
        group_oh = jax.nn.one_hot(group_idx, G)            # (B, T, G)
        # Einsum selects the group_idx-th slice along the G axis per token
        selected_logits = jnp.einsum(
            "btg,btgm->btm", group_oh, expert_logits_grouped
        )  # (B, T, M)

        # Pick top-k within selected group
        top_k_vals, intra_idx = jax.lax.top_k(selected_logits, K)  # (B, T, K)
        gates = jax.nn.softmax(top_k_vals, axis=-1)                  # (B, T, K)

        # Convert intra-group indices → global expert indices
        # global = intra + group_idx * M
        global_indices = intra_idx + group_idx[..., None] * M        # (B, T, K)

        # ---- Group-level load-balancing auxiliary loss ----
        N = B * T
        flat_group_idx = group_idx.reshape(N)                        # (N,)
        group_dispatch = jax.nn.one_hot(flat_group_idx, G)           # (N, G)
        f_g = jnp.mean(group_dispatch, axis=0)                       # (G,)
        P_g = jnp.mean(group_probs.reshape(N, G), axis=0)            # (G,)
        group_aux = G * jnp.sum(f_g * P_g)

        # ---- Expert-level load-balancing auxiliary loss ----
        expert_probs = jax.nn.softmax(expert_logits, axis=-1)         # (B, T, E)
        flat_global_top1 = global_indices.reshape(N, K)[:, 0]        # (N,)
        expert_dispatch = jax.nn.one_hot(flat_global_top1, E)        # (N, E)
        f_e = jnp.mean(expert_dispatch, axis=0)                      # (E,)
        P_e = jnp.mean(expert_probs.reshape(N, E), axis=0)           # (E,)
        expert_aux = E * jnp.sum(f_e * P_e)

        aux_loss = group_aux + expert_aux

        return gates, global_indices, aux_loss, z_loss


# ---------------------------------------------------------------------------
# Hierarchical MoE Layer
# ---------------------------------------------------------------------------

class HierMoELayer(nn.Module):
    """Capacity-aware Hierarchical MoE layer.

    Uses :class:`HierRouter` for two-stage (group → expert) routing, then
    the same static-shape token dispatch / collect pattern as
    :class:`MoELayer` for JIT-friendly sparse execution.

    Requires ``config.n_expert_groups > 1`` and
    ``config.n_experts % config.n_expert_groups == 0``.
    """

    config: NanoMoEConfig

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model)
            deterministic: If True, disable dropout and jitter noise.

        Returns:
            output:       (batch, seq_len, d_model)
            combined_aux: Scalar — aux load-balance loss + z_loss_coeff * z_loss.
        """
        cfg = self.config
        B, T, D = x.shape
        N = B * T
        E, K, G = cfg.n_experts, cfg.top_k, cfg.n_expert_groups

        # ---- Hierarchical Router ----
        gates, indices, aux_loss, z_loss = HierRouter(
            n_expert_groups=G, n_experts=E, top_k=K,
            jitter_noise=cfg.router_jitter_noise,
        )(x, deterministic=deterministic)
        gates_flat = gates.reshape(N, K)      # (N, K)
        indices_flat = indices.reshape(N, K)  # (N, K)

        # ---- Capacity per expert ----
        capacity = max(1, math.ceil(cfg.capacity_factor * N / E))

        # ---- Build dispatch tensor (same as MoELayer) ----
        expert_mask = jax.nn.one_hot(indices_flat, E)           # (N, K, E)
        flat_mask   = expert_mask.reshape(N * K, E)             # (N*K, E)
        cumcounts   = jnp.cumsum(flat_mask, axis=0)             # (N*K, E)
        slot_indices = jnp.sum(
            (cumcounts.reshape(N, K, E) - 1) * expert_mask, axis=-1
        ).astype(jnp.int32)  # (N, K)

        capacity_mask  = slot_indices < capacity                 # (N, K)
        effective_gates = jnp.where(capacity_mask, gates_flat, 0.0)

        safe_slots = jnp.clip(slot_indices, 0, capacity - 1)    # (N, K)
        slot_oh    = jax.nn.one_hot(safe_slots, capacity)        # (N, K, C)
        dispatch   = jnp.einsum(
            "nke,nkc->ecn",
            expert_mask * capacity_mask[..., None],
            slot_oh,
        )  # (E, C, N)

        # ---- Gather → run experts → collect ----
        tokens       = x.reshape(N, D)                           # (N, D)
        expert_input = jnp.einsum("ecn,nd->ecd", dispatch, tokens)  # (E, C, D)

        experts = [
            ExpertFFN(d_ff=cfg.d_ff, d_model=D, name=f"expert_{i}")
            for i in range(E)
        ]
        expert_outputs = jnp.stack(
            [experts[i](expert_input[i]) for i in range(E)], axis=0
        )  # (E, C, D)

        per_expert_output = jnp.einsum(
            "ecn,ecd->end", dispatch, expert_outputs
        )  # (E, N, D)

        per_expert_gate = jnp.einsum(
            "nk,nke->ne", effective_gates, expert_mask
        )  # (N, E)
        output_flat = jnp.einsum(
            "ne,end->nd", per_expert_gate, per_expert_output
        )  # (N, D)

        output = output_flat.reshape(B, T, D)
        output = nn.Dropout(rate=cfg.dropout_rate)(output, deterministic=deterministic)

        combined_aux = aux_loss + cfg.z_loss_coeff * z_loss
        return output, combined_aux


# ---------------------------------------------------------------------------
# Hierarchical Transformer Block
# ---------------------------------------------------------------------------

class HierTransformerBlock(nn.Module):
    """Pre-norm transformer block using :class:`HierMoELayer` instead of
    the flat :class:`MoELayer`."""

    config: NanoMoEConfig

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Self-attention sub-layer
        residual = x
        x = nn.LayerNorm()(x)
        x = MultiHeadAttention(config=self.config)(x, deterministic=deterministic)
        x = x + residual

        # HierMoE sub-layer
        residual = x
        x_norm = nn.LayerNorm()(x)
        moe_out, aux_loss = HierMoELayer(config=self.config)(
            x_norm, deterministic=deterministic
        )
        x = moe_out + residual

        return x, aux_loss
