"""Validate request association and hybrid-cache slot isolation."""
def validate_cache_implementation(audit):
    """Check cache mechanics at matched batch size, not cross-batch argmax."""
    import math
    assert audit["input_tokens_exact"], "wrong input token/request association"
    for field in ("prestep_state_max_abs_error",
                  "mixed_age_vs_same_age_batch_max_abs_error",
                  "mixed_age_vs_same_age_rollout_state_max_abs_error"):
        assert audit[field] and all(math.isfinite(v) and v == 0.0 for v in audit[field].values()), field
    assert audit["mixed_age_vs_same_age_rollout_tokens_exact"], "matched-batch rollout mismatch"
    recycle = audit["slot_recycle_isolation"]
    for field in ("recycled_row_vs_fresh_prefix_max_abs_error", "untouched_rows_max_abs_error"):
        assert recycle[field] and all(math.isfinite(v) and v == 0.0 for v in recycle[field].values()), field
    assert audit["after_ages"] == [age + 1 for age in audit["ages"]]
    assert audit["rollout_after_ages"] == [age + audit["rollout_steps"] for age in audit["ages"]]
