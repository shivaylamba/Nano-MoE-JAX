"""Tests for nano_moe.layers — Router, Expert, MoE, Attention."""

import jax
import jax.numpy as jnp
import pytest

from nano_moe.config import NanoMoEConfig
from nano_moe.layers import ExpertFFN, Router, MoELayer, MultiHeadAttention, TransformerBlock


@pytest.fixture
def config():
    return NanoMoEConfig(
        vocab_size=64,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=64,
        n_experts=4,
        top_k=2,
        block_size=16,
        dropout_rate=0.0,
        capacity_factor=1.25,
        router_jitter_noise=0.0,
        z_loss_coeff=1e-3,
    )


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def dummy_input(config, rng):
    """Random float input (B, T, D)."""
    return jax.random.normal(rng, (2, 8, config.d_model))


# ----- ExpertFFN -----

class TestExpertFFN:
    def test_output_shape(self, config, rng, dummy_input):
        model = ExpertFFN(d_ff=config.d_ff, d_model=config.d_model)
        params = model.init(rng, dummy_input[0, 0])["params"]  # single vector
        out = model.apply({"params": params}, dummy_input)
        assert out.shape == dummy_input.shape


# ----- Router -----

class TestRouter:
    def test_output_shapes(self, config, rng, dummy_input):
        model = Router(n_experts=config.n_experts, top_k=config.top_k)
        params = model.init(rng, dummy_input, deterministic=True)["params"]
        gates, indices, aux_loss, z_loss = model.apply(
            {"params": params}, dummy_input, deterministic=True
        )

        B, T, _ = dummy_input.shape
        assert gates.shape == (B, T, config.top_k)
        assert indices.shape == (B, T, config.top_k)
        assert aux_loss.shape == ()  # scalar
        assert z_loss.shape == ()   # scalar

    def test_gates_sum_to_one(self, config, rng, dummy_input):
        model = Router(n_experts=config.n_experts, top_k=config.top_k)
        params = model.init(rng, dummy_input, deterministic=True)["params"]
        gates, _, _, _ = model.apply({"params": params}, dummy_input, deterministic=True)

        gate_sums = jnp.sum(gates, axis=-1)
        assert jnp.allclose(gate_sums, 1.0, atol=1e-5)

    def test_aux_loss_positive(self, config, rng, dummy_input):
        model = Router(n_experts=config.n_experts, top_k=config.top_k)
        params = model.init(rng, dummy_input, deterministic=True)["params"]
        _, _, aux_loss, _ = model.apply({"params": params}, dummy_input, deterministic=True)

        assert float(aux_loss) > 0.0

    def test_z_loss_positive(self, config, rng, dummy_input):
        model = Router(n_experts=config.n_experts, top_k=config.top_k)
        params = model.init(rng, dummy_input, deterministic=True)["params"]
        _, _, _, z_loss = model.apply({"params": params}, dummy_input, deterministic=True)

        assert float(z_loss) > 0.0

    def test_jitter_noise_changes_gates(self, config, rng, dummy_input):
        """Gates should differ between deterministic=True and False when jitter_noise > 0."""
        model = Router(n_experts=config.n_experts, top_k=config.top_k, jitter_noise=0.1)
        params = model.init(rng, dummy_input, deterministic=True)["params"]

        gates_det, _, _, _ = model.apply(
            {"params": params}, dummy_input, deterministic=True
        )
        gates_noisy, _, _, _ = model.apply(
            {"params": params}, dummy_input, deterministic=False,
            rngs={"dropout": jax.random.PRNGKey(42)},
        )

        # With noise the gate values should differ from the noiseless run
        assert not jnp.allclose(gates_det, gates_noisy, atol=1e-6)


# ----- MoELayer -----

class TestMoELayer:
    def test_output_shape(self, config, rng, dummy_input):
        model = MoELayer(config=config)
        params = model.init(rng, dummy_input)["params"]
        output, aux_loss = model.apply({"params": params}, dummy_input, deterministic=True)

        assert output.shape == dummy_input.shape
        assert aux_loss.shape == ()

    def test_aux_loss_nonzero(self, config, rng, dummy_input):
        model = MoELayer(config=config)
        params = model.init(rng, dummy_input)["params"]
        _, aux_loss = model.apply({"params": params}, dummy_input, deterministic=True)

        assert float(aux_loss) > 0.0

    def test_capacity_masking_no_nans(self, config, rng, dummy_input):
        """Capacity-limited routing must not produce NaN/Inf outputs and must
        still yield non-trivial (non-zero) values for surviving tokens."""
        # Use a tiny capacity_factor to force heavy dropping
        tight_cfg = NanoMoEConfig(
            vocab_size=64, n_layers=2, n_heads=2, d_model=32, d_ff=64,
            n_experts=4, top_k=2, block_size=16, dropout_rate=0.0,
            capacity_factor=0.5,  # aggressively small → many tokens dropped
        )
        model = MoELayer(config=tight_cfg)
        params = model.init(rng, dummy_input)["params"]
        output, _ = model.apply({"params": params}, dummy_input, deterministic=True)

        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))
        # Some tokens survive; the output norm must be strictly positive
        assert float(jnp.sum(jnp.abs(output))) > 0.0


# ----- MultiHeadAttention -----

class TestMultiHeadAttention:
    def test_output_shape(self, config, rng, dummy_input):
        model = MultiHeadAttention(config=config)
        params = model.init(rng, dummy_input)["params"]
        out = model.apply({"params": params}, dummy_input, deterministic=True)

        assert out.shape == dummy_input.shape


# ----- TransformerBlock -----

class TestTransformerBlock:
    def test_output_shape(self, config, rng, dummy_input):
        model = TransformerBlock(config=config)
        params = model.init(rng, dummy_input)["params"]
        out, aux_loss = model.apply({"params": params}, dummy_input, deterministic=True)

        assert out.shape == dummy_input.shape
        assert aux_loss.shape == ()
