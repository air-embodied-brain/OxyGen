import asyncio
import http
import logging
import time
import traceback
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


def _has_attrs(obj: Any, names: list[str]) -> bool:
    return all(hasattr(obj, name) for name in names)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        infer_api: str = "infer",
        continuous_batching_kwargs: dict[str, Any] | None = None,
        continuous_batching_request_mode: str = "resume_until_finished",
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._requested_infer_api = infer_api
        self._infer_api = infer_api
        self._continuous_batching_kwargs = continuous_batching_kwargs or {}
        if continuous_batching_request_mode not in ("resume_until_finished", "new_each_call"):
            raise ValueError(f"Unknown continuous batching request mode: {continuous_batching_request_mode}")
        self._continuous_batching_request_mode = continuous_batching_request_mode
        logging.getLogger("websockets.server").setLevel(logging.INFO)

        supports_shared_kv = _has_attrs(
            self._policy,
            ["infer_text_actions_shared_kv", "_sample_text_actions_shared_kv"],
        )
        if self._infer_api == "shared_kv" and not supports_shared_kv:
            raise RuntimeError("requested infer_api=shared_kv but policy lacks shared_kv capability")

        supports_continuous_batching = _has_attrs(
            self._policy,
            [
                "infer_text_actions_continuous_batch",
                "init_continuous_batching",
                "_prefill",
                "_init_incremental_state",
                "_generate_n_tokens",
                "_sample_actions_with_kv",
            ],
        )
        if self._infer_api == "continuous_batching" and not supports_continuous_batching:
            raise RuntimeError(
                "requested infer_api=continuous_batching but policy lacks required incremental-kv methods"
            )

        self._cache_manager = None
        if self._infer_api == "continuous_batching":
            self._cache_manager = self._policy.init_continuous_batching()

        self._metadata["requested_infer_api"] = self._requested_infer_api
        self._metadata["effective_infer_api"] = self._infer_api
        if self._infer_api == "continuous_batching":
            self._metadata["continuous_batching_request_mode"] = self._continuous_batching_request_mode
        logger.info("Websocket infer api: %s", self._infer_api)

    @staticmethod
    def _normalize_actions(action: dict[str, Any]) -> dict[str, Any]:
        actions = np.asarray(action["actions"])
        if actions.ndim == 1:
            actions = actions[None, :]
        action["actions"] = actions
        return action

    @staticmethod
    def _language_update(result: dict[str, Any], *, created_this_call: bool) -> dict[str, Any]:
        return {
            "request_id": result.get("request_id"),
            "tokens_this_frame": result.get("tokens_this_frame"),
            "tokens_full": result.get("tokens_full"),
            "text": result.get("text", ""),
            "is_finished": bool(result.get("is_finished", False)),
            "created_this_call": created_this_call,
        }

    def _infer_once(
        self,
        obs: dict[str, Any],
        request_state: str | list[str] | None,
    ) -> tuple[dict[str, Any], str | list[str] | None]:
        if self._infer_api == "infer":
            return self._policy.infer(obs), request_state

        if self._infer_api == "shared_kv":
            return self._policy.infer_text_actions_shared_kv(obs), request_state

        if self._infer_api == "continuous_batching":
            if self._continuous_batching_request_mode == "new_each_call":
                active_request_ids = list(request_state or [])
                request_ids = [None, *active_request_ids]
                results = self._policy.infer_text_actions_continuous_batch(
                    [obs] * len(request_ids),
                    cache_manager=self._cache_manager,
                    request_ids=request_ids,
                    generate_actions_for_resumed=False,
                    **self._continuous_batching_kwargs,
                )
                action = results[0]
                action["language_updates"] = [
                    self._language_update(result, created_this_call=index == 0) for index, result in enumerate(results)
                ]
                next_active_request_ids = [
                    result["request_id"] for result in results if not result.get("is_finished", False)
                ]
                action["active_language_requests"] = len(next_active_request_ids)
                return action, next_active_request_ids

            request_id = request_state if isinstance(request_state, str) else None
            results = self._policy.infer_text_actions_continuous_batch(
                [obs],
                cache_manager=self._cache_manager,
                request_ids=[request_id],
                generate_actions_for_resumed=True,
                **self._continuous_batching_kwargs,
            )
            action = results[0]
            next_request_id = action.get("request_id")
            if action.get("is_finished", False):
                next_request_id = None
            return action, next_request_id

        raise ValueError(f"Unsupported infer_api: {self._infer_api}")

    def _remove_request_state(self, request_state: str | list[str] | None) -> None:
        if self._cache_manager is None or request_state is None:
            return
        request_ids = [request_state] if isinstance(request_state, str) else request_state
        for request_id in request_ids:
            self._cache_manager.remove_state(request_id)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        request_state: str | list[str] | None = (
            []
            if self._infer_api == "continuous_batching" and self._continuous_batching_request_mode == "new_each_call"
            else None
        )

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action, request_state = self._infer_once(obs, request_state)
                action = self._normalize_actions(action)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                    "infer_api": self._infer_api,
                    "requested_infer_api": self._requested_infer_api,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                self._remove_request_state(request_state)
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                self._remove_request_state(request_state)
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
