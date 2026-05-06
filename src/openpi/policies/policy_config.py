import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    requested_checkpoint = str(checkpoint_dir)
    # If the caller passed a path that doesn't exist locally AND isn't a
    # remote URI that maybe_download knows how to fetch, fall back to random
    # weights. This is only meant for quick benchmarking / plumbing tests
    # where actual model outputs don't matter.
    is_remote_uri = "://" in requested_checkpoint
    use_random_init = (not is_remote_uri) and (not os.path.exists(requested_checkpoint))

    if use_random_init:
        banner = "*" * 72
        logging.warning(
            "\n%s\n"
            "DUMMY MODE: checkpoint path %r does not exist.\n"
            "Initializing the model with RANDOM WEIGHTS — outputs are garbage\n"
            "and only shapes / latency / memory behavior are meaningful.\n"
            "%s",
            banner, requested_checkpoint, banner,
        )
        # Pick the backend: explicit pytorch_device wins, otherwise look for
        # a "pytorch" hint in the requested path (so existing conventions like
        # .../pi05_base_pytorch keep working), otherwise default to JAX.
        is_pytorch = (pytorch_device is not None) or ("pytorch" in requested_checkpoint.lower())
        checkpoint_dir = pathlib.Path(requested_checkpoint)
        logging.warning(
            "DUMMY MODE: using %s backend.", "pytorch" if is_pytorch else "jax",
        )
    else:
        checkpoint_dir = download.maybe_download(requested_checkpoint)
        weight_path = os.path.join(checkpoint_dir, "model.safetensors")
        is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if use_random_init:
        if is_pytorch:
            model = _init_random_pytorch_model(train_config)
        else:
            import jax
            model = train_config.model.create(jax.random.key(0))
    elif is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params(train_config.pytorch_training_precision)
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        # norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            # transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            # transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )


def _init_random_pytorch_model(train_config: _config.TrainConfig):
    """Build the PyTorch model with its default (random) parameter init.

    Mirrors ``BaseModelConfig.load_pytorch`` but skips the safetensors load,
    so no checkpoint file is required. Intended for dummy / smoke-test runs.
    """
    import torch

    from openpi.models import model as _model_mod
    from openpi.models_pytorch import pi0_pytorch
    from openpi.models_pytorch import pi05_pytorch

    if train_config.model.model_type is _model_mod.ModelType.PI05_O2:
        model = pi05_pytorch.PI05Pytorch(config=train_config.model)
    else:
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
    if train_config.pytorch_training_precision == "bfloat16":
        model = model.to(torch.bfloat16)
    elif train_config.pytorch_training_precision == "float32":
        model = model.to(torch.float32)
    else:
        raise ValueError(f"Unsupported PyTorch precision: {train_config.pytorch_training_precision}")
    return model
