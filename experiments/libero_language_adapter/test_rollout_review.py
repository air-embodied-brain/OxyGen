from experiments.libero_language_adapter import rollout_review


def _update(request_id: str, slots: int, *, created: bool = False, finished: bool = False) -> dict:
    return {
        "request_id": request_id,
        "text": f"request {request_id} at {slots}",
        "is_finished": finished,
        "tokens_full": list(range(slots)),
        "tokens_this_frame": list(range(max(0, slots - 5), slots)),
        "created_this_call": created,
    }


def test_request_buffer_preserves_creation_order_while_requests_advance() -> None:
    request_buffer = []
    rollout_review._apply_request_updates(  # noqa: SLF001
        request_buffer, [_update("req_0", 5, created=True)], step=10
    )
    rollout_review._apply_request_updates(  # noqa: SLF001
        request_buffer,
        [_update("req_1", 5, created=True), _update("req_0", 10)],
        step=15,
    )
    rollout_review._apply_request_updates(  # noqa: SLF001
        request_buffer,
        [_update("req_2", 5, created=True), _update("req_1", 10), _update("req_0", 15, finished=True)],
        step=20,
    )

    assert [request["request_id"] for request in request_buffer] == ["req_2", "req_1", "req_0"]
    assert [request["start_step"] for request in request_buffer] == [20, 15, 10]
    assert [request["token_count"] for request in request_buffer] == [5, 10, 15]
    assert [request["is_finished"] for request in request_buffer] == [False, False, True]
