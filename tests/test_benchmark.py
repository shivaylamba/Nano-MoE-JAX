"""Architecture benchmark: NanoMoE baseline vs NanoMoE++ vs HierMoE.

Compares three configurations on three axes:
  - **Latency**    – median forward-pass wall-clock time (ms) after JIT warm-up.
  - **Convergence** – cross-entropy loss after a fixed number of training steps.
  - **Expert balance** – combined auxiliary loss (lower = more balanced routing).

Run as a pytest module for CI assertions, or as a standalone script for a
human-readable results table:

    python -m pytest tests/test_benchmark.py -v
    python tests/test_benchmark.py
"""

import time
from typing import Tuple, Type

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nano_moe.config import NanoMoEConfig
from nano_moe.model import NanoMoE, HierNanoMoE
from nano_moe.train import create_train_state, train_step
from nano_moe.utils import get_batch, count_params


# ---------------------------------------------------------------------------
# Benchmark configs — all use the same small dimensions so CI finishes fast
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

# Version A — NanoMoE (original flat routing, no capacity limit or aux losses)
BASELINE_CFG = NanoMoEConfig(
    **_BASE_KWARGS,
    capacity_factor=100.0,    # effectively unlimited — no token dropping
    router_jitter_noise=0.0,
    z_loss_coeff=0.0,
    aux_loss_coeff=0.01,
    n_expert_groups=1,
)

# Version B — NanoMoE++ (capacity-aware dispatch, z-loss, jitter noise)
PLUSPLUS_CFG = NanoMoEConfig(
    **_BASE_KWARGS,
    capacity_factor=1.25,
    router_jitter_noise=0.1,
    z_loss_coeff=1e-3,
    aux_loss_coeff=0.01,
    n_expert_groups=1,
)

# Version C — HierMoE (two-stage hierarchical routing: group → expert)
# n_experts=4, n_expert_groups=2 → 2 groups of 2 experts.
# With top_k=2 and experts_per_group=2, both experts in the selected group
# are activated, giving the same active-expert budget as the flat variants.
# For top_k < experts_per_group, fewer experts would activate per token.
HIERMOE_CFG = NanoMoEConfig(
    **_BASE_KWARGS,
    capacity_factor=1.25,
    router_jitter_noise=0.1,
    z_loss_coeff=1e-3,
    aux_loss_coeff=0.01,
    n_expert_groups=2,        # enables HierMoE two-stage routing
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
    model,
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
    cfg: NanoMoEConfig,
    rng: jax.Array,
    model_cls: Type = NanoMoE,
    steps: int = _TRAIN_STEPS,
) -> Tuple[float, float]:
    """Train for `steps` steps; return (initial_ce_loss, final_ce_loss)."""
    data_rng, init_rng, loop_rng = jax.random.split(rng, 3)
    data = _make_data(data_rng, cfg, n=cfg.block_size * 40)
    state = create_train_state(init_rng, cfg, model_cls=model_cls)

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
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def benchmark_rng():
    return jax.random.PRNGKey(123)


@pytest.fixture(scope="module")
def baseline_model_params(benchmark_rng):
    _, init_rng = jax.random.split(benchmark_rng)
    model = NanoMoE(config=BASELINE_CFG)
    dummy = jnp.ones((1, BASELINE_CFG.block_size), dtype=jnp.int32)
    params = model.init(init_rng, dummy, deterministic=True)["params"]
    return model, params


@pytest.fixture(scope="module")
def plusplus_model_params(benchmark_rng):
    _, init_rng = jax.random.split(benchmark_rng)
    model = NanoMoE(config=PLUSPLUS_CFG)
    dummy = jnp.ones((1, PLUSPLUS_CFG.block_size), dtype=jnp.int32)
    params = model.init(init_rng, dummy, deterministic=True)["params"]
    return model, params


@pytest.fixture(scope="module")
def hiermoe_model_params(benchmark_rng):
    _, init_rng = jax.random.split(benchmark_rng)
    model = HierNanoMoE(config=HIERMOE_CFG)
    dummy = jnp.ones((1, HIERMOE_CFG.block_size), dtype=jnp.int32)
    params = model.init(init_rng, dummy, deterministic=True)["params"]
    return model, params


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestLatency:
    """All three variants must produce finite latencies; overhead bounded."""

    def test_baseline_latency_finite(self, baseline_model_params):
        model, params = baseline_model_params
        x = jnp.ones((1, BASELINE_CFG.block_size), dtype=jnp.int32)
        lat = _measure_latency_ms(model, params, x)
        assert lat > 0.0 and not np.isnan(lat)

    def test_plusplus_latency_finite(self, plusplus_model_params):
        model, params = plusplus_model_params
        x = jnp.ones((1, PLUSPLUS_CFG.block_size), dtype=jnp.int32)
        lat = _measure_latency_ms(model, params, x)
        assert lat > 0.0 and not np.isnan(lat)

    def test_hiermoe_latency_finite(self, hiermoe_model_params):
        model, params = hiermoe_model_params
        x = jnp.ones((1, HIERMOE_CFG.block_size), dtype=jnp.int32)
        lat = _measure_latency_ms(model, params, x)
        assert lat > 0.0 and not np.isnan(lat)

    def test_hiermoe_latency_overhead_bounded(
        self, baseline_model_params, hiermoe_model_params
    ):
        """HierMoE overhead (two routers) must be < 20× the baseline."""
        bmodel, bparams = baseline_model_params
        hmodel, hparams = hiermoe_model_params
        x = jnp.ones((1, BASELINE_CFG.block_size), dtype=jnp.int32)
        lat_b = _measure_latency_ms(bmodel, bparams, x)
        lat_h = _measure_latency_ms(hmodel, hparams, x)
        ratio = lat_h / lat_b
        assert ratio < 20.0, (
            f"HierMoE latency is {ratio:.1f}× baseline; expected < 20×"
        )


class TestConvergence:
    """All three variants must show loss decrease during training."""

    def test_baseline_loss_decreases(self, benchmark_rng):
        _, sub = jax.random.split(benchmark_rng)
        init_ce, final_ce = _run_training(BASELINE_CFG, sub, NanoMoE)
        assert final_ce < init_ce, "Baseline CE loss did not decrease"

    def test_plusplus_loss_decreases(self, benchmark_rng):
        _, sub = jax.random.split(benchmark_rng)
        init_ce, final_ce = _run_training(PLUSPLUS_CFG, sub, NanoMoE)
        assert final_ce < init_ce, "NanoMoE++ CE loss did not decrease"

    def test_hiermoe_loss_decreases(self, benchmark_rng):
        _, sub = jax.random.split(benchmark_rng)
        init_ce, final_ce = _run_training(HIERMOE_CFG, sub, HierNanoMoE)
        assert final_ce < init_ce, "HierMoE CE loss did not decrease"

    def test_hiermoe_ce_comparable_to_baseline(self, benchmark_rng):
        """HierMoE final CE must be within 2× of the baseline."""
        rng1, rng2 = jax.random.split(benchmark_rng)
        _, ce_b = _run_training(BASELINE_CFG, rng1, NanoMoE)
        _, ce_h = _run_training(HIERMOE_CFG, rng2, HierNanoMoE)
        ratio = ce_h / ce_b
        assert ratio < 2.0, (
            f"HierMoE final CE ({ce_h:.3f}) is {ratio:.2f}× baseline ({ce_b:.3f})"
        )

    def test_plusplus_ce_comparable_to_baseline(self, benchmark_rng):
        """NanoMoE++ final CE must be within 2× of the baseline."""
        rng1, rng2 = jax.random.split(benchmark_rng)
        _, ce_b = _run_training(BASELINE_CFG, rng1, NanoMoE)
        _, ce_p = _run_training(PLUSPLUS_CFG, rng2, NanoMoE)
        ratio = ce_p / ce_b
        assert ratio < 2.0, (
            f"NanoMoE++ final CE ({ce_p:.3f}) is {ratio:.2f}× baseline ({ce_b:.3f})"
        )


class TestExpertBalance:
    """Aux loss must be positive for all variants (routing is active)."""

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

    def test_hiermoe_aux_loss_positive(self, hiermoe_model_params):
        model, params = hiermoe_model_params
        x = jnp.ones((2, HIERMOE_CFG.block_size), dtype=jnp.int32)
        _, aux = model.apply({"params": params}, x, deterministic=True)
        assert float(aux) > 0.0


class TestParamCount:
    """Baseline and NanoMoE++ must share the same parameter count (architecture
    identical).  HierMoE must have strictly more parameters (extra group router)."""

    def test_baseline_plusplus_same_param_count(
        self, baseline_model_params, plusplus_model_params
    ):
        _, bparams = baseline_model_params
        _, pparams = plusplus_model_params
        assert count_params(bparams) == count_params(pparams)

    def test_hiermoe_has_more_params_than_baseline(
        self, baseline_model_params, hiermoe_model_params
    ):
        _, bparams = baseline_model_params
        _, hparams = hiermoe_model_params
        assert count_params(hparams) > count_params(bparams)


# ---------------------------------------------------------------------------
# Standalone runner — prints a 3-way comparison table
# ---------------------------------------------------------------------------

def run_comparison() -> None:
    """Run the full benchmark and print a human-readable 3-way results table."""
    rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, 7)

    arch_configs = [
        ("NanoMoE (baseline)", BASELINE_CFG, NanoMoE,     keys[0], keys[1]),
        ("NanoMoE++",          PLUSPLUS_CFG, NanoMoE,     keys[2], keys[3]),
        ("HierMoE",            HIERMOE_CFG,  HierNanoMoE, keys[4], keys[5]),
    ]

    rows = []
    for label, cfg, model_cls, init_rng, train_rng in arch_configs:
        print(f"  Benchmarking {label} ...", flush=True)

        model = model_cls(config=cfg)
        dummy = jnp.ones((1, cfg.block_size), dtype=jnp.int32)
        params = model.init(init_rng, dummy, deterministic=True)["params"]
        n_params = count_params(params)

        x_lat = jnp.ones((1, cfg.block_size), dtype=jnp.int32)
        lat_ms = _measure_latency_ms(model, params, x_lat)

        init_ce, final_ce = _run_training(cfg, train_rng, model_cls, steps=_TRAIN_STEPS)

        x_eval = jnp.ones((2, cfg.block_size), dtype=jnp.int32)
        _, aux = model.apply({"params": params}, x_eval, deterministic=True)

        rows.append({
            "label":      label,
            "params":     n_params,
            "latency_ms": lat_ms,
            "init_ce":    init_ce,
            "final_ce":   final_ce,
            "ce_drop":    init_ce - final_ce,
            "aux_loss":   float(aux),
        })

    # ---- Print table ----
    cols = [
        ("Architecture",  "label",      "s",   20),
        ("Params",        "params",     ",d",   9),
        ("Latency (ms)",  "latency_ms", ".3f", 13),
        ("Init CE",       "init_ce",    ".4f",  9),
        ("Final CE",      "final_ce",   ".4f",  9),
        ("CE ↓",          "ce_drop",    ".4f",  8),
        ("Aux loss",      "aux_loss",   ".4f",  9),
    ]

    divider    = "+" + "+".join("-" * (w + 2) for _, _, _, w in cols) + "+"
    header     = "|" + "|".join(f" {t:<{w}} " for t, _, _, w in cols) + "|"
    table_width = len(divider)

    print()
    print("=" * table_width)
    print("  3-Way Architecture Comparison: NanoMoE vs NanoMoE++ vs HierMoE")
    print(f"  Training steps: {_TRAIN_STEPS}   Batch: {_BATCH_SIZE}   "
          f"d_model={_BASE_KWARGS['d_model']}  n_experts={_BASE_KWARGS['n_experts']}  "
          f"n_groups(HierMoE)={HIERMOE_CFG.n_expert_groups}")
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

    # ---- Delta summary ----
    if len(rows) == 3:
        b, p, h = rows
        print()
        print("  Relative to NanoMoE (baseline):")
        for other in (p, h):
            lat_ratio = other["latency_ms"] / b["latency_ms"]
            ce_ratio  = other["final_ce"] / b["final_ce"]
            ce_dir    = "better" if other["final_ce"] < b["final_ce"] else "worse"
            print(
                f"    {other['label']:<20}  "
                f"latency {lat_ratio:.2f}×   "
                f"final CE {ce_ratio:.3f}× ({ce_dir})   "
                f"params +{other['params'] - b['params']:,}"
            )
        print()


if __name__ == "__main__":
    print("Running 3-way architecture comparison benchmark …\n")
    run_comparison()

