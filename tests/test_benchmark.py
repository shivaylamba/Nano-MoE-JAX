"""Architecture benchmark: NanoMoE baseline vs NanoMoE++.

Compares the two configurations on three axes:
  - **Latency**       – median forward-pass wall-clock time (ms) after JIT warm-up.
  - **Convergence**   – cross-entropy loss after a fixed number of training steps.
  - **Expert balance** – variance of per-expert token-fraction (lower = more balanced).

Run as a pytest module for CI assertions, or as a standalone script for a
human-readable results table:

    python -m pytest tests/test_benchmark.py -v
    python tests/test_benchmark.py
"""

import time
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nano_moe.config import NanoMoEConfig
from nano_moe.model import NanoMoE
from nano_moe.train import create_train_state, train_step
from nano_moe.utils import get_batch, count_params


# ---------------------------------------------------------------------------
# Shared benchmark config – small enough to run in CI
# ---------------------------------------------------------------------------

_BASE_KWARGS = dict(
    vocab_size=64,
    n_layers=2,
    n_heads=2,
    d_model=32,
    d_ff=64,
    n_experts=4,
    top_k=2,
    block_size=16,
    dropout_rate=0.0,
    learning_rate=3e-4,
    weight_decay=0.1,
    batch_size=4,
    max_iters=10,
    eval_interval=10,
    eval_iters=2,
)

BASELINE_CFG = NanoMoEConfig(
    **_BASE_KWARGS,
    # Baseline (original NanoMoE style): no capacity limit, no z-loss, no jitter
    capacity_factor=100.0,
    router_jitter_noise=0.0,
    z_loss_coeff=0.0,
    aux_loss_coeff=0.01,
)

PLUSPLUS_CFG = NanoMoEConfig(
    **_BASE_KWARGS,
    # NanoMoE++: capacity-aware, z-loss, jitter noise
    capacity_factor=1.25,
    router_jitter_noise=0.1,
    z_loss_coeff=1e-3,
    aux_loss_coeff=0.01,
)

_WARMUP = 3        # JIT warm-up iterations before timing
_MEASURE = 15      # timing iterations for median latency
_TRAIN_STEPS = 50  # training steps for convergence tests
_BATCH_SIZE = 4    # batch size used during convergence tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(rng: jax.Array, cfg: NanoMoEConfig, n: int = 512) -> np.ndarray:
    """Synthetic 1-D random token array."""
    return np.array(
        jax.random.randint(rng, (n,), 0, cfg.vocab_size), dtype=np.int32
    )


def _measure_latency_ms(
    model: NanoMoE,
    params,
    x: jnp.ndarray,
    warmup: int = _WARMUP,
    steps: int = _MEASURE,
) -> float:
    """Median forward-pass latency in milliseconds (after JIT warm-up)."""
    jit_fn = jax.jit(
        lambda p, inp: model.apply({"params": p}, inp, deterministic=True)
    )
    for _ in range(warmup):
        out = jit_fn(params, x)
        out[0].block_until_ready()

    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        out = jit_fn(params, x)
        out[0].block_until_ready()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def _run_training(
    cfg: NanoMoEConfig, rng: jax.Array, steps: int = _TRAIN_STEPS
) -> Tuple[float, float]:
    """Train for `steps` steps; return (initial_ce_loss, final_ce_loss)."""
    data_rng, init_rng, loop_rng = jax.random.split(rng, 3)
    data = _make_data(data_rng, cfg, n=cfg.block_size * 40)
    state = create_train_state(init_rng, cfg)

    initial_ce: float = float("nan")
    final_ce: float = float("nan")

    for i in range(steps):
        loop_rng, batch_rng = jax.random.split(loop_rng)
        x, y = get_batch(data, _BATCH_SIZE, cfg.block_size, batch_rng)
        state, metrics = train_step(state, x, y, cfg)
        ce = float(metrics["ce_loss"])
        if i == 0:
            initial_ce = ce
        final_ce = ce

    return initial_ce, final_ce


# ---------------------------------------------------------------------------
# Pytest tests (CI assertions)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def benchmark_rng():
    return jax.random.PRNGKey(123)


@pytest.fixture(scope="module")
def baseline_model_params(benchmark_rng):
    rng, init_rng = jax.random.split(benchmark_rng)
    model = NanoMoE(config=BASELINE_CFG)
    dummy = jnp.ones((1, BASELINE_CFG.block_size), dtype=jnp.int32)
    params = model.init(init_rng, dummy, deterministic=True)["params"]
    return model, params


@pytest.fixture(scope="module")
def plusplus_model_params(benchmark_rng):
    rng, init_rng = jax.random.split(benchmark_rng)
    model = NanoMoE(config=PLUSPLUS_CFG)
    dummy = jnp.ones((1, PLUSPLUS_CFG.block_size), dtype=jnp.int32)
    params = model.init(init_rng, dummy, deterministic=True)["params"]
    return model, params


class TestLatency:
    """NanoMoE++ forward latency must be within 10× of the baseline."""

    def test_baseline_latency_finite(self, baseline_model_params):
        model, params = baseline_model_params
        x = jnp.ones((1, BASELINE_CFG.block_size), dtype=jnp.int32)
        lat = _measure_latency_ms(model, params, x)
        assert lat > 0.0
        assert not np.isnan(lat)

    def test_plusplus_latency_finite(self, plusplus_model_params):
        model, params = plusplus_model_params
        x = jnp.ones((1, PLUSPLUS_CFG.block_size), dtype=jnp.int32)
        lat = _measure_latency_ms(model, params, x)
        assert lat > 0.0
        assert not np.isnan(lat)

    def test_latency_overhead_bounded(self, baseline_model_params, plusplus_model_params):
        """NanoMoE++ dispatch overhead must be less than 10× the baseline."""
        bmodel, bparams = baseline_model_params
        pmodel, pparams = plusplus_model_params
        x = jnp.ones((1, BASELINE_CFG.block_size), dtype=jnp.int32)
        lat_b = _measure_latency_ms(bmodel, bparams, x)
        lat_p = _measure_latency_ms(pmodel, pparams, x)
        ratio = lat_p / lat_b
        assert ratio < 10.0, (
            f"NanoMoE++ latency is {ratio:.1f}× baseline; expected < 10×"
        )


class TestConvergence:
    """NanoMoE++ training must show loss decrease and comparable final CE."""

    def test_baseline_loss_decreases(self, benchmark_rng):
        rng, sub = jax.random.split(benchmark_rng)
        init_ce, final_ce = _run_training(BASELINE_CFG, sub, steps=_TRAIN_STEPS)
        assert final_ce < init_ce, "Baseline CE loss did not decrease"

    def test_plusplus_loss_decreases(self, benchmark_rng):
        rng, sub = jax.random.split(benchmark_rng)
        init_ce, final_ce = _run_training(PLUSPLUS_CFG, sub, steps=_TRAIN_STEPS)
        assert final_ce < init_ce, "NanoMoE++ CE loss did not decrease"

    def test_plusplus_ce_comparable_to_baseline(self, benchmark_rng):
        """Final CE loss of NanoMoE++ should be within 2× of the baseline."""
        rng1, rng2 = jax.random.split(benchmark_rng)
        _, ce_b = _run_training(BASELINE_CFG, rng1, steps=_TRAIN_STEPS)
        _, ce_p = _run_training(PLUSPLUS_CFG, rng2, steps=_TRAIN_STEPS)
        ratio = ce_p / ce_b
        assert ratio < 2.0, (
            f"NanoMoE++ final CE ({ce_p:.3f}) is {ratio:.2f}× baseline ({ce_b:.3f})"
        )


class TestExpertBalance:
    """NanoMoE++ aux loss must be positive (routing is active)."""

    def test_baseline_aux_loss_positive(self, baseline_model_params):
        model, params = baseline_model_params
        x = jnp.ones((2, BASELINE_CFG.block_size), dtype=jnp.int32)
        _, aux = model.apply({"params": params}, x, deterministic=True)
        assert float(aux) > 0.0

    def test_plusplus_aux_loss_positive(self, plusplus_model_params):
        model, params = plusplus_model_params
        x = jnp.ones((2, PLUSPLUS_CFG.block_size), dtype=jnp.int32)
        _, aux = model.apply({"params": params}, x, deterministic=True)
        assert float(aux) > 0.0


class TestParamCount:
    """Both configs should have the same parameter count (architecture unchanged)."""

    def test_same_param_count(self, baseline_model_params, plusplus_model_params):
        _, bparams = baseline_model_params
        _, pparams = plusplus_model_params
        assert count_params(bparams) == count_params(pparams)


# ---------------------------------------------------------------------------
# Standalone runner — prints a comparison table
# ---------------------------------------------------------------------------

def run_comparison() -> None:
    """Run the full benchmark and print a human-readable results table."""
    import sys

    rng = jax.random.PRNGKey(0)
    rng, rng_b_init, rng_p_init, rng_b_train, rng_p_train = jax.random.split(rng, 5)

    configs = [
        ("NanoMoE (baseline)", BASELINE_CFG, rng_b_init, rng_b_train),
        ("NanoMoE++",          PLUSPLUS_CFG, rng_p_init, rng_p_train),
    ]

    rows = []
    for label, cfg, init_rng, train_rng in configs:
        print(f"  Benchmarking {label} ...", flush=True)

        # Init model
        model = NanoMoE(config=cfg)
        dummy = jnp.ones((1, cfg.block_size), dtype=jnp.int32)
        params = model.init(init_rng, dummy, deterministic=True)["params"]
        n_params = count_params(params)

        # Latency
        x_lat = jnp.ones((1, cfg.block_size), dtype=jnp.int32)
        lat_ms = _measure_latency_ms(model, params, x_lat)

        # Convergence
        init_ce, final_ce = _run_training(cfg, train_rng, steps=_TRAIN_STEPS)

        # Aux loss at eval
        x_eval = jnp.ones((2, cfg.block_size), dtype=jnp.int32)
        _, aux = model.apply({"params": params}, x_eval, deterministic=True)

        rows.append({
            "label": label,
            "params": n_params,
            "latency_ms": lat_ms,
            "init_ce": init_ce,
            "final_ce": final_ce,
            "ce_drop": init_ce - final_ce,
            "aux_loss": float(aux),
        })

    # Print table
    cols = [
        ("Architecture",     "label",      "s",   22),
        ("Params",           "params",     ",d",   9),
        ("Latency (ms)",     "latency_ms", ".3f", 14),
        ("Init CE",          "init_ce",    ".4f", 10),
        ("Final CE",         "final_ce",   ".4f", 10),
        ("CE ↓",             "ce_drop",    ".4f",  8),
        ("Aux loss",         "aux_loss",   ".4f", 10),
    ]

    divider = "+" + "+".join("-" * (w + 2) for _, _, _, w in cols) + "+"
    header  = "|" + "|".join(f" {t:<{w}} " for t, _, _, w in cols) + "|"
    table_width = len(divider)

    print()
    print("=" * table_width)
    print("  NanoMoE vs NanoMoE++ — Benchmark Results")
    print(f"  Training steps: {_TRAIN_STEPS}   Batch: {_BATCH_SIZE}   "
          f"d_model: {_BASE_KWARGS['d_model']}   Experts: {_BASE_KWARGS['n_experts']}")
    print("=" * table_width)
    print(divider)
    print(header)
    print(divider)
    for r in rows:
        row = "|" + "|".join(
            f" {format(r[k], fmt):<{w}} " for _, k, fmt, w in cols
        ) + "|"
        print(row)
    print(divider)

    # Delta row
    if len(rows) == 2:
        b, p = rows[0], rows[1]
        print()
        print(f"  Latency overhead    : {p['latency_ms'] / b['latency_ms']:.2f}×  "
              f"({p['latency_ms']:.3f} ms vs {b['latency_ms']:.3f} ms)")
        print(f"  Final CE ratio      : {p['final_ce'] / b['final_ce']:.3f}×  "
              f"({p['final_ce']:.4f} vs {b['final_ce']:.4f})")
        print(f"  CE improvement (↓)  : {b['final_ce'] - p['final_ce']:+.4f}  "
              f"({'NanoMoE++ better' if p['final_ce'] < b['final_ce'] else 'Baseline better'})")
        print()


if __name__ == "__main__":
    print("Running NanoMoE vs NanoMoE++ comparison benchmark …\n")
    run_comparison()
