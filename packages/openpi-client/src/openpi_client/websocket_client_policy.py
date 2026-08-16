import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._last_transport_timing: Dict[str, float | int] = {}
        self._language_token_histories: Dict[str, list[int]] = {}
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def get_last_transport_timing(self) -> Dict[str, float | int]:
        return dict(self._last_transport_timing)

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def _expand_language_update_deltas(self, result: Dict) -> Dict:
        if self._server_metadata.get("language_update_schema") != "token_delta_v1":
            return result

        for update in result.get("language_updates", []):
            request_id = str(update["request_id"])
            if update.get("created_this_call", False):
                self._language_token_histories.pop(request_id, None)
            history = self._language_token_histories.setdefault(request_id, [])
            delta = np.asarray(update.get("tokens_this_frame", []), dtype=np.int32)
            history.extend(delta.tolist())
            expected_count = int(update.get("token_count", len(history)))
            if len(history) != expected_count:
                raise RuntimeError(
                    f"Language token delta mismatch for {request_id}: reconstructed {len(history)}, "
                    f"expected {expected_count}"
                )
            update["tokens_full"] = np.asarray(history, dtype=np.int32)
            if update.get("is_finished", False):
                self._language_token_histories.pop(request_id, None)
        return result

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        pack_started = time.perf_counter()
        data = self._packer.pack(obs)
        pack_finished = time.perf_counter()
        self._ws.send(data)
        send_finished = time.perf_counter()
        response = self._ws.recv()
        receive_finished = time.perf_counter()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        result = self._expand_language_update_deltas(msgpack_numpy.unpackb(response))
        unpack_finished = time.perf_counter()
        self._last_transport_timing = {
            "request_pack_ms": (pack_finished - pack_started) * 1000,
            "request_send_ms": (send_finished - pack_finished) * 1000,
            "response_wait_ms": (receive_finished - send_finished) * 1000,
            "response_unpack_ms": (unpack_finished - receive_finished) * 1000,
            "request_bytes": len(data),
            "response_bytes": len(response),
            "round_trip_ms": (unpack_finished - pack_started) * 1000,
        }
        return result

    @override
    def reset(self) -> None:
        self._language_token_histories.clear()
