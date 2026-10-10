"""Fixed-input, matched-batch checks of Xiaomi continuous-cache mechanics.

No task-quality or greedy-token equivalence requirement. Checks are untimed.
"""
import copy
import torch
from oxygen_runtime import (prefill_vlm, init_language_from_prefix,
                            staticize_language_state, batched_language_steps,
                            StaticLanguageCache)

ATOL = 1e-4
RTOL = 1e-3  # Matched shapes/history, not a bound on cross-batch BF16 variation.


def tensor_check(actual, expected):
    if actual.shape != expected.shape:
        return False, float('inf')
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return False, float('inf')
    error = float((actual.float() - expected.float()).abs().max())
    return bool(torch.allclose(actual.float(), expected.float(), atol=ATOL, rtol=RTOL)), error


def cache_check(actual, expected):
    if (actual.prefix_length, actual.max_decode_steps) != (expected.prefix_length, expected.max_decode_steps):
        return False, float('inf')
    pairs = list(zip(actual.key_cache, expected.key_cache)) + list(zip(actual.value_cache, expected.value_cache))
    if len(actual.key_cache) != len(expected.key_cache) or len(actual.value_cache) != len(expected.value_cache):
        return False, float('inf')
    checks = [tensor_check(a, b) for a, b in pairs]
    return bool(checks) and all(ok for ok, _ in checks), max((err for _, err in checks), default=float('inf'))


@torch.inference_mode()
def implementation_audit(model, processor, inputs, steps=30):
    ages = [0, 1, 3, 5, 8]
    capacity = max(ages) + steps + 2
    prefix = prefill_vlm(model, dict(inputs))
    vocab_size = int(prefix.outputs.logits.shape[-1])
    def fresh(row):
        state = init_language_from_prefix(processor, prefix, stop_on_eos=False)
        staticize_language_state(state, capacity)
        # Distinct, fixed request histories expose row swaps without argmax drift.
        state.forced_ids = [(1000 + row * 97 + offset * 13) % vocab_size for offset in range(capacity)]
        return state

    snapshots = []
    for row, age in enumerate(ages):
        state = fresh(row)
        if age:batched_language_steps(model, [state], age)
        snapshots.append(state)
    splits = StaticLanguageCache.from_batch_splits([s.cache for s in snapshots]).batch_split()
    roundtrip_ok = all(cache_check(c, s.cache)[0] for c, s in zip(splits, snapshots))
    # Negative control: replacing a row by another age/history must be detected.
    row_swap_detected = not cache_check(snapshots[0].cache, snapshots[1].cache)[0]
    mixed = copy.deepcopy(snapshots)
    controls = [[copy.deepcopy(state) for _ in ages] for state in snapshots]
    observed = {}
    def capture(module, args, output):
        observed['logits'] = output.logits[:, -1].detach().float().clone()
    hook = model.vlm.register_forward_hook(capture)
    records = []
    recycle_ok = True
    try:
        for step in range(steps):
            if step == steps // 2:
                # A departing request is replaced by a fresh one; others stay live.
                row = 2
                before = [copy.deepcopy(s.cache) for s in mixed]
                mixed[row] = fresh(row + len(ages))
                controls[row] = [copy.deepcopy(mixed[row]) for _ in ages]
                recycle_ok &= all(cache_check(s.cache, old)[0] for i, (s, old) in enumerate(zip(mixed, before)) if i != row)
            expected_tokens = [s.forced_ids[s.forced_index] for s in mixed]
            lengths = [len(s.generated_ids) for s in mixed]
            emitted = batched_language_steps(model, mixed, 1)
            logits = observed['logits'].clone()
            for row, group in enumerate(controls):
                batched_language_steps(model, group, 1)
                logits_ok, logits_error = tensor_check(logits[row], observed['logits'][0])
                cache_ok, cache_error = cache_check(mixed[row].cache, group[0].cache)
                progress_ok = (emitted[row] == [expected_tokens[row]] and len(mixed[row].generated_ids) == lengths[row] + 1
                               and mixed[row].generated_ids == group[0].generated_ids)
                records.append(dict(step=step, row=row, logits_close=logits_ok,
                                    logits_max_abs=logits_error, cache_close=cache_ok,
                                    cache_max_abs=cache_error, progress_correct=progress_ok))
    finally:
        hook.remove()
    passed = (roundtrip_ok and row_swap_detected and recycle_ok and len(records) == steps * len(ages)
              and all(x['logits_close'] and x['cache_close'] and x['progress_correct'] for x in records))
    return dict(policy='fixed_input_matched_batch_v2', ages=ages, steps=steps,
                atol=ATOL, rtol=RTOL, stack_split_roundtrip=roundtrip_ok,
                negative_row_swap_detected=row_swap_detected, recycle_isolation=recycle_ok,
                records=records, passed=passed)
