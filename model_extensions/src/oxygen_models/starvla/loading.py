"""Construct StarVLA experts from metadata without loading trained weights."""

import json
from pathlib import Path
from threading import RLock

import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, Qwen3VLForConditionalGeneration
from starVLA.model.framework.base_framework import build_framework, merge_config_overrides
from starVLA.model.framework.share_tools import apply_config_compat, dict_to_namespace, read_mode_config
from starVLA.model.modules.vlm import QWen3

_CONSTRUCTION_LOCK = RLock()


class _ConfigOnlyQwen3:
    @staticmethod
    def from_pretrained(model_id, **kwargs):
        config = AutoConfig.from_pretrained(model_id, local_files_only=True)
        config._attn_implementation = kwargs.get("attn_implementation", "sdpa")
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(kwargs.get("dtype", torch.bfloat16))
            return Qwen3VLForConditionalGeneration(config)
        finally:
            torch.set_default_dtype(previous_dtype)


def from_config_only(metadata_checkpoint, config_overrides=None):
    """Load an expert's config and schema, initializing its parameters locally.

    The upstream wrapper constructs its backbone through ``from_pretrained``.
    During construction, substitute a config-only factory in that wrapper's
    module; restore it even if construction fails. Upstream files stay unchanged.
    Call during setup, before starting inference workers.
    """
    path = Path(metadata_checkpoint)
    if path.is_dir():
        config = OmegaConf.load(path / "config.yaml")
        apply_config_compat(config)
        model_config = OmegaConf.to_container(config, resolve=True)
        norm_stats = json.loads((path / "dataset_statistics.json").read_text())
    else:
        model_config, norm_stats = read_mode_config(path)
    config = dict_to_namespace(merge_config_overrides(model_config, config_overrides))
    if "Qwen3-VL" not in config.framework.qwenvl.base_vlm:
        raise ValueError("Config-only expert loading requires a Qwen3-VL backbone")
    config.trainer.pretrained_checkpoint = None
    with _CONSTRUCTION_LOCK:
        original = QWen3.Qwen3VLForConditionalGeneration
        try:
            QWen3.Qwen3VLForConditionalGeneration = _ConfigOnlyQwen3
            model = build_framework(cfg=config)
        finally:
            QWen3.Qwen3VLForConditionalGeneration = original
    model.norm_stats = norm_stats
    return model
