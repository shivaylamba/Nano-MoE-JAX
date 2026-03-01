"""Tests for nano_moe.model — NanoMoE forward, loss, generate; HierNanoMoE."""

import jax
import jax.numpy as jnp
import pytest

from nano_moe.config import NanoMoEConfig
from nano_moe.model import NanoMoE, HierNanoMoE
from nano_moe.utils import count_params


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
def hier_config():
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
        n_expert_groups=2,
    )


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def model_and_params(config, rng):
    model = NanoMoE(config=config)
    dummy = jnp.ones((1, config.block_size), dtype=jnp.int32)
    params = model.init(rng, dummy, deterministic=True)["params"]
    return model, params


@pytest.fixture
def hier_model_and_params(hier_config, rng):
    model = HierNanoMoE(config=hier_config)
    dummy = jnp.ones((1, hier_config.block_size), dtype=jnp.int32)
    params = model.init(rng, dummy, deterministic=True)["params"]
    return model, params


# ----- Forward pass -----

class TestNanoMoE:
    def test_logits_shape(self, config, model_and_params):
        model, params = model_and_params
        x = jnp.ones((2, 8), dtype=jnp.int32)
        logits, aux_loss = model.apply({"params": params}, x, deterministic=True)

        assert logits.shape == (2, 8, config.vocab_size)
        assert aux_loss.shape == ()

    def test_aux_loss_positive(self, config, model_and_params):
        model, params = model_and_params
        x = jnp.ones((2, 8), dtype=jnp.int32)
        _, aux_loss = model.apply({"params": params}, x, deterministic=True)

        assert float(aux_loss) > 0.0

    def test_parameter_count(self, model_and_params):
        _, params = model_and_params
        n = count_params(params)
        assert n > 0

    def test_generate_length(self, config, model_and_params, rng):
        model, params = model_and_params
        prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)
        new_tokens = 10

        tokens = model.generate(
            params, rng, prompt,
            max_new_tokens=new_tokens, temperature=1.0, top_k_sample=0,
        )

        assert tokens.shape == (1, 3 + new_tokens)

    def test_no_nan_in_logits(self, config, model_and_params):
        model, params = model_and_params
        x = jnp.ones((1, config.block_size), dtype=jnp.int32)
        logits, _ = model.apply({"params": params}, x, deterministic=True)

        assert not jnp.any(jnp.isnan(logits))
        assert not jnp.any(jnp.isinf(logits))


# ---------------------------------------------------------------------------
# HierNanoMoE
# ---------------------------------------------------------------------------

class TestHierNanoMoE:
    def test_logits_shape(self, hier_config, hier_model_and_params):
        model, params = hier_model_and_params
        x = jnp.ones((2, 8), dtype=jnp.int32)
        logits, aux_loss = model.apply({"params": params}, x, deterministic=True)

        assert logits.shape == (2, 8, hier_config.vocab_size)
        assert aux_loss.shape == ()

    def test_aux_loss_positive(self, hier_config, hier_model_and_params):
        model, params = hier_model_and_params
        x = jnp.ones((2, 8), dtype=jnp.int32)
        _, aux_loss = model.apply({"params": params}, x, deterministic=True)

        assert float(aux_loss) > 0.0

    def test_parameter_count(self, hier_model_and_params):
        _, params = hier_model_and_params
        n = count_params(params)
        assert n > 0

    def test_generate_length(self, hier_config, hier_model_and_params, rng):
        model, params = hier_model_and_params
        prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)
        new_tokens = 10

        tokens = model.generate(
            params, rng, prompt,
            max_new_tokens=new_tokens, temperature=1.0, top_k_sample=0,
        )

        assert tokens.shape == (1, 3 + new_tokens)

    def test_no_nan_in_logits(self, hier_config, hier_model_and_params):
        model, params = hier_model_and_params
        x = jnp.ones((1, hier_config.block_size), dtype=jnp.int32)
        logits, _ = model.apply({"params": params}, x, deterministic=True)

        assert not jnp.any(jnp.isnan(logits))
        assert not jnp.any(jnp.isinf(logits))

    def test_hier_has_more_params_than_flat(self, model_and_params, hier_model_and_params):
        """HierNanoMoE adds a group router; it should have more parameters than NanoMoE."""
        _, flat_params = model_and_params
        _, hier_params = hier_model_and_params
        assert count_params(hier_params) > count_params(flat_params)
