import numpy as np

from openpi_client import websocket_client_policy


def test_expand_language_update_deltas_reconstructs_full_history() -> None:
    client = websocket_client_policy.WebsocketClientPolicy.__new__(websocket_client_policy.WebsocketClientPolicy)
    client._server_metadata = {"language_update_schema": "token_delta_v1"}  # noqa: SLF001
    client._language_token_histories = {}  # noqa: SLF001

    first = client._expand_language_update_deltas(  # noqa: SLF001
        {
            "language_updates": [
                {
                    "request_id": "req_0",
                    "tokens_this_frame": np.asarray([1, 2], dtype=np.int32),
                    "token_count": 2,
                    "created_this_call": True,
                    "is_finished": False,
                }
            ]
        }
    )
    second = client._expand_language_update_deltas(  # noqa: SLF001
        {
            "language_updates": [
                {
                    "request_id": "req_0",
                    "tokens_this_frame": np.asarray([3, 4], dtype=np.int32),
                    "token_count": 4,
                    "created_this_call": False,
                    "is_finished": True,
                }
            ]
        }
    )

    np.testing.assert_array_equal(first["language_updates"][0]["tokens_full"], [1, 2])
    np.testing.assert_array_equal(second["language_updates"][0]["tokens_full"], [1, 2, 3, 4])
    assert client._language_token_histories == {}  # noqa: SLF001
