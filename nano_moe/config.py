"""Hyperparameter configuration for NanoMoE."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NanoMoEConfig:
    """All hyperparameters for the NanoMoE model.

    Attributes:
        vocab_size: Size of the token vocabulary (character-level by default).
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads.
        d_model: Hidden / embedding dimension.
        d_ff: Inner dimension of expert feed-forward networks.
        n_experts: Total number of expert FFNs in each MoE layer.
        top_k: Number of experts activated per token.
        block_size: Maximum sequence length (context window).
        capacity_factor: Expert capacity multiplier. Each expert holds at most
            ``ceil(capacity_factor * tokens / n_experts)`` tokens per forward
            pass; assignments beyond this limit are dropped (zero-gated).
            Under perfectly uniform routing, each expert receives
            ``tokens * top_k / n_experts`` assignments, so
            ``capacity_factor >= top_k`` guarantees no dropping.
        router_jitter_noise: Half-width of uniform additive noise applied to
            router logits during training (0 = disabled).  Improves exploration
            and reduces expert collapse.
        dropout_rate: Dropout probability (used during training).
        aux_loss_coeff: Weight of the load-balancing auxiliary loss.
        z_loss_coeff: Weight of the z-loss that penalises large router logit
            magnitudes and keeps routing distributions stable.
        learning_rate: Peak learning rate for AdamW.
        weight_decay: AdamW weight decay coefficient.
        batch_size: Training batch size.
        max_iters: Maximum training iterations.
        eval_interval: Iterations between evaluation runs.
        eval_iters: Number of batches used for evaluation.
    """

    # --- Model architecture ---
    vocab_size: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    n_experts: int = 4
    top_k: int = 2
    block_size: int = 128

    # --- MoE routing ---
    capacity_factor: float = 1.25
    router_jitter_noise: float = 0.0

    # --- Regularization ---
    dropout_rate: float = 0.1
    aux_loss_coeff: float = 0.01
    z_loss_coeff: float = 1e-3

    # --- Optimiser ---
    learning_rate: float = 3e-4
    weight_decay: float = 0.1

    # --- Training schedule ---
    batch_size: int = 32
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50
