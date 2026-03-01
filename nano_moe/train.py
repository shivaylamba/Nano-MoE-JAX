"""Training utilities: train state, JIT-compiled steps, and training loop."""

from typing import Any, Dict, Tuple, Type
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from nano_moe.config import NanoMoEConfig
from nano_moe.model import NanoMoE


class TrainState(train_state.TrainState):
    """Extended TrainState carrying a dropout PRNG key."""

    dropout_rng: jax.Array


def create_train_state(
    rng: jax.Array,
    config: NanoMoEConfig,
    model_cls: Type = NanoMoE,
) -> TrainState:
    """Initialise model and optimiser.

    Args:
        rng: PRNG key.
        config: Model/training hyperparameters.
        model_cls: Model class to instantiate (default :class:`~nano_moe.model.NanoMoE`).
            Pass :class:`~nano_moe.model.HierNanoMoE` to train the hierarchical model.

    Returns:
        A TrainState with initialised parameters and AdamW optimiser.
    """
    model = model_cls(config=config)

    rng, init_rng, dropout_rng = jax.random.split(rng, 3)

    # Dummy input for parameter initialisation
    dummy = jnp.ones((1, config.block_size), dtype=jnp.int32)
    params = model.init({"params": init_rng, "dropout": dropout_rng}, dummy, deterministic=False)[
        "params"
    ]

    # AdamW with optional learning-rate warmup could be added here
    tx = optax.adamw(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dropout_rng=dropout_rng,
    )


@partial(jax.jit, static_argnums=(3,))
def train_step(
    state: TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    config: NanoMoEConfig,
) -> Tuple[TrainState, Dict[str, Any]]:
    """Single JIT-compiled training step.

    Args:
        state: Current training state.
        x: Input tokens (batch, seq_len).
        y: Target tokens (batch, seq_len).
        config: Hyperparameters (static — used for aux_loss_coeff).

    Returns:
        Updated state and a metrics dict {"loss", "ce_loss", "aux_loss"}.
    """
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def loss_fn(params):
        logits, aux_loss = state.apply_fn(
            {"params": params},
            x,
            deterministic=False,
            rngs={"dropout": dropout_rng},
        )
        # Cross-entropy loss
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        ce_loss = jnp.mean(ce_loss)

        total_loss = ce_loss + config.aux_loss_coeff * aux_loss
        return total_loss, (ce_loss, aux_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (ce_loss, aux_loss)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(dropout_rng=new_dropout_rng)

    metrics = {"loss": loss, "ce_loss": ce_loss, "aux_loss": aux_loss}
    return state, metrics


@partial(jax.jit, static_argnums=(3,))
def eval_step(
    state: TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    config: NanoMoEConfig,
) -> Dict[str, Any]:
    """JIT-compiled evaluation step (no gradient, no dropout).

    Returns:
        Metrics dict {"loss", "ce_loss", "aux_loss"}.
    """
    logits, aux_loss = state.apply_fn(
        {"params": state.params},
        x,
        deterministic=True,
    )
    ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    ce_loss = jnp.mean(ce_loss)
    loss = ce_loss + config.aux_loss_coeff * aux_loss

    return {"loss": loss, "ce_loss": ce_loss, "aux_loss": aux_loss}


def train(
    state: TrainState,
    config: NanoMoEConfig,
    train_data: jnp.ndarray,
    val_data: jnp.ndarray,
    rng: jax.Array,
) -> TrainState:
    """Main training loop with periodic evaluation.

    Args:
        state: Initial training state.
        config: Hyperparameters.
        train_data: 1-D integer array of training tokens.
        val_data: 1-D integer array of validation tokens.
        rng: PRNG key for batch sampling.

    Returns:
        Final training state.
    """
    from nano_moe.utils import get_batch  # avoid circular import

    best_val_loss = float("inf")

    for step in range(1, config.max_iters + 1):
        rng, batch_rng = jax.random.split(rng)

        # ---------- Training step ----------
        x, y = get_batch(train_data, config.batch_size, config.block_size, batch_rng)
        state, metrics = train_step(state, x, y, config)

        # ---------- Evaluation ----------
        if step % config.eval_interval == 0 or step == 1:
            val_losses = []
            for i in range(config.eval_iters):
                rng, eval_rng = jax.random.split(rng)
                vx, vy = get_batch(val_data, config.batch_size, config.block_size, eval_rng)
                val_metrics = eval_step(state, vx, vy, config)
                val_losses.append(float(val_metrics["loss"]))

            avg_val = sum(val_losses) / len(val_losses)
            marker = ""
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                marker = " ★"

            print(
                f"[step {step:>5d}/{config.max_iters}]  "
                f"train loss: {float(metrics['loss']):.4f}  "
                f"(ce: {float(metrics['ce_loss']):.4f}, aux: {float(metrics['aux_loss']):.4f})  |  "
                f"val loss: {avg_val:.4f}{marker}"
            )

    return state
