from __future__ import annotations

import numpy as np

from openpi.serving import websocket_policy_server


class _FakeCacheManager:
    def __init__(self) -> None:
        self.active_states: dict[str, object] = {}
        self.removed: list[str] = []

    def remove_state(self, request_id: str) -> None:
        self.active_states.pop(request_id, None)
        self.removed.append(request_id)


class _FakeContinuousBatchingPolicy:
    def __init__(self) -> None:
        self.manager = _FakeCacheManager()
        self.calls: list[dict] = []
        self.next_request_id = 0

    def init_continuous_batching(self) -> _FakeCacheManager:
        return self.manager

    def _prefill(self) -> None:
        pass

    def _init_incremental_state(self) -> None:
        pass

    def _generate_n_tokens(self) -> None:
        pass

    def _sample_actions_with_kv(self) -> None:
        pass

    def infer_text_actions_continuous_batch(
        self,
        obs_list: list[dict],
        *,
        cache_manager: _FakeCacheManager,
        request_ids: list[str | None],
        generate_actions_for_resumed: bool,
        **kwargs,
    ) -> list[dict]:
        self.calls.append(
            {
                "obs_count": len(obs_list),
                "request_ids": list(request_ids),
                "generate_actions_for_resumed": generate_actions_for_resumed,
                "kwargs": kwargs,
            }
        )
        results = []
        for index, candidate_request_id in enumerate(request_ids):
            request_id = candidate_request_id
            if request_id is None:
                request_id = f"req_{self.next_request_id}"
                self.next_request_id += 1
                decoded = 5
            else:
                decoded = int(cache_manager.active_states[request_id]) + 5
            is_finished = decoded >= 15
            if is_finished:
                cache_manager.remove_state(request_id)
            else:
                cache_manager.active_states[request_id] = decoded
            results.append(
                {
                    "actions": np.ones((10, 7), dtype=np.float32) if index == 0 else None,
                    "request_id": request_id,
                    "tokens_this_frame": np.arange(5),
                    "tokens_full": np.arange(decoded),
                    "text": f"decoded {decoded}",
                    "is_finished": is_finished,
                    "policy_timing": {
                        "batch_size": len(request_ids),
                        "new_requests": 1,
                        "resumed_requests": len(request_ids) - 1,
                    },
                }
            )
        return results


def test_new_each_call_batches_all_unfinished_language_requests() -> None:
    policy = _FakeContinuousBatchingPolicy()
    server = websocket_policy_server.WebsocketPolicyServer(
        policy,
        infer_api="continuous_batching",
        continuous_batching_request_mode="new_each_call",
        continuous_batching_kwargs={"steps_per_frame": 5, "max_decoding_steps": 15},
    )

    request_state: list[str] = []
    responses = []
    for frame in range(3):
        response, next_state = server._infer_once({"frame": frame}, request_state)  # noqa: SLF001
        assert isinstance(next_state, list)
        request_state = next_state
        responses.append(response)

    assert [call["obs_count"] for call in policy.calls] == [1, 2, 3]
    assert [call["request_ids"] for call in policy.calls] == [
        [None],
        [None, "req_0"],
        [None, "req_1", "req_0"],
    ]
    assert all(not call["generate_actions_for_resumed"] for call in policy.calls)
    assert [response["policy_timing"]["batch_size"] for response in responses] == [1, 2, 3]
    assert [response["active_language_requests"] for response in responses] == [1, 2, 2]
    assert [len(response["language_updates"]) for response in responses] == [1, 2, 3]
    assert all(response["actions"].shape == (10, 7) for response in responses)
    assert request_state == ["req_2", "req_1"]
    assert "req_0" not in policy.manager.active_states


def test_remove_request_state_cleans_up_all_active_requests() -> None:
    policy = _FakeContinuousBatchingPolicy()
    server = websocket_policy_server.WebsocketPolicyServer(
        policy,
        infer_api="continuous_batching",
        continuous_batching_request_mode="new_each_call",
    )
    policy.manager.active_states.update({"req_2": 5, "req_1": 10})

    server._remove_request_state(["req_2", "req_1"])  # noqa: SLF001

    assert policy.manager.active_states == {}
    assert policy.manager.removed == ["req_2", "req_1"]
