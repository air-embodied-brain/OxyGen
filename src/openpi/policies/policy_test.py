import jax
import jax.numpy as jnp
from openpi_client import action_chunk_broker
import pytest

from openpi.models.pi05 import IncrementalTextState
from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class _ModelWithAdapterMethod:
    def __init__(self, language_adapter: str):
        self.language_adapter = language_adapter

    def init_language_adapter_incremental_state(self):
        pass


def test_language_adapter_detection_uses_model_config():
    assert not _policy._uses_suffix_language_adapter(_ModelWithAdapterMethod("none"))  # noqa: SLF001
    assert _policy._uses_suffix_language_adapter(_ModelWithAdapterMethod("suffix_lora"))  # noqa: SLF001


def _incremental_state(value: int, *, suffix_offset: int = 2) -> IncrementalTextState:
    return IncrementalTextState(
        rng=jax.random.split(jax.random.key(value), 1),
        last_logits=jnp.full((1, 1, 4), value, dtype=jnp.float32),
        output_tokens=jnp.full((1, 3), value, dtype=jnp.int32),
        kv_cache=(
            jnp.full((2, 1), value, dtype=jnp.int32),
            jnp.full((2, 1, 5, 1, 1), value, dtype=jnp.float32),
            jnp.full((2, 1, 5, 1, 1), value, dtype=jnp.float32),
        ),
        current_step=jnp.asarray([value], dtype=jnp.int32),
        is_finished=jnp.asarray([False]),
        prefill_len=jnp.asarray([3], dtype=jnp.int32),
        prefill_size=3,
        suffix_offset=suffix_offset,
        max_decoding_steps=3,
        cache_size=5,
    )


def test_incremental_state_stack_split_preserves_suffix_offset():
    states = [_incremental_state(0), _incremental_state(1)]

    restored = _policy._split_incremental_state(  # noqa: SLF001
        _policy._stack_incremental_states(states),
        2,  # noqa: SLF001
    )

    assert [state.suffix_offset for state in restored] == [2, 2]
    for expected, actual in zip(states, restored, strict=True):
        assert jax.tree.all(jax.tree.map(jnp.array_equal, expected, actual))


def test_incremental_state_stack_rejects_different_suffix_offsets():
    with pytest.raises(ValueError, match="suffix_offset"):
        _policy._stack_incremental_states(  # noqa: SLF001
            [_incremental_state(0), _incremental_state(1, suffix_offset=3)]
        )


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
