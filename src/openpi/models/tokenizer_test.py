import numpy as np

from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_tokenize_language_suffix():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    inputs, targets, mask, loss_mask = tokenizer.tokenize_language_suffix(
        "Subtask: ", "Put the bowl on the plate.", max_len=16
    )

    assert inputs.shape == targets.shape == mask.shape == loss_mask.shape == (16,)
    assert np.all(loss_mask <= mask)
    assert loss_mask.sum() > 1
    first_loss = int(np.flatnonzero(loss_mask)[0])
    assert inputs[first_loss] == tokenizer.tokenize_language_seed("Subtask: ")[-1]
    assert tokenizer.eos_token_id in targets[loss_mask]


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)
    print(action - act)


if __name__ == "__main__":
    test_tokenize()
    test_fast_tokenizer()
