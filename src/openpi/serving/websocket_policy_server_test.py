import numpy as np

from openpi.serving import websocket_policy_server


class _CacheManager:
    def __init__(self):
        self.active_states = {}
        self.removed = []

    def remove_state(self, request_id):
        self.active_states.pop(request_id, None)
        self.removed.append(request_id)


class _ContinuousPolicy:
    def __init__(self):
        self.manager = _CacheManager()
        self.calls = []
        self.next_request_id = 0

    def init_continuous_batching(self):
        return self.manager

    def infer_text_actions_continuous_batch(
        self,
        obs_list,
        *,
        cache_manager,
        request_ids,
        generate_actions_for_resumed,
        **kwargs,
    ):
        self.calls.append(list(request_ids))
        results = []
        for index, existing_request_id in enumerate(request_ids):
            request_id = existing_request_id
            if request_id is None:
                request_id = f"req_{self.next_request_id}"
                self.next_request_id += 1
                decoded = 2
            else:
                decoded = cache_manager.active_states[request_id] + 2
            finished = decoded >= 6
            if finished:
                cache_manager.remove_state(request_id)
            else:
                cache_manager.active_states[request_id] = decoded
            results.append(
                {
                    "actions": np.ones((10, 7)) if index == 0 else None,
                    "request_id": request_id,
                    "tokens_this_frame": np.arange(2),
                    "tokens_full": np.arange(decoded),
                    "text": f"decoded {decoded}",
                    "is_finished": finished,
                }
            )
        return results


class _BlockingPolicy:
    def infer_text_actions_blocking_baseline(self, obs, **kwargs):
        return {
            "actions": np.ones((10, 7)),
            "tokens_this_frame": np.arange(3),
            "tokens_full": np.arange(3),
            "text": "Memory complete.",
            "is_finished": True,
        }


def test_new_request_is_batched_with_unfinished_requests():
    policy = _ContinuousPolicy()
    server = websocket_policy_server.WebsocketPolicyServer(
        policy,
        infer_api="continuous_batching",
        continuous_batching_request_mode="new_each_call",
        inference_kwargs={"steps_per_frame": 2},
    )

    request_state = []
    responses = []
    for frame in range(3):
        response, request_state = server._infer_once({"frame": frame}, request_state)  # noqa: SLF001
        responses.append(response)

    assert policy.calls == [[None], [None, "req_0"], [None, "req_1", "req_0"]]
    assert [len(response["language_updates"]) for response in responses] == [1, 2, 3]
    assert request_state == ["req_2", "req_1"]
    assert responses[-1]["language_updates"][-1]["is_finished"]

    server._remove_request_state(request_state)  # noqa: SLF001
    assert policy.manager.active_states == {}


def test_blocking_baseline_returns_one_completed_update():
    server = websocket_policy_server.WebsocketPolicyServer(
        _BlockingPolicy(),
        infer_api="blocking_baseline",
        inference_kwargs={"max_decoding_steps": 28},
    )

    response, request_state = server._infer_once({"frame": 0}, None)  # noqa: SLF001

    assert request_state is None
    assert response["actions"].shape == (10, 7)
    update = response["language_updates"][0]
    assert update["text"] == "Memory complete."
    assert update["is_finished"]
    assert update["created_this_call"]
    np.testing.assert_array_equal(update["tokens_full"], response["tokens_full"])
