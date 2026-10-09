"""Check that config-only construction cannot leak into pretrained loading."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3VLConfig
from oxygen_models.starvla import loading


class LoadingTest(unittest.TestCase):
    def test_factory_preserves_parameters_and_dtype(self):
        config = Qwen3VLConfig(
            text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]}),
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64,
                               num_heads=4, out_hidden_size=32, deepstack_visual_indexes=[]),
        )
        original_dtype = torch.get_default_dtype()
        torch.manual_seed(0)
        try:
            torch.set_default_dtype(torch.bfloat16)
            expected = loading.Qwen3VLForConditionalGeneration(config)
        finally:
            torch.set_default_dtype(original_dtype)
        torch.manual_seed(0)
        with patch.object(loading.AutoConfig, 'from_pretrained', return_value=config):
            actual = loading._ConfigOnlyQwen3.from_pretrained('local', dtype=torch.bfloat16)
        self.assertEqual(torch.get_default_dtype(), original_dtype)
        for name, value in expected.state_dict().items():
            self.assertTrue(torch.equal(value, actual.state_dict()[name]), name)

    def test_restores_factory_on_success_and_failure(self):
        original = loading.QWen3.Qwen3VLForConditionalGeneration
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'config.yaml').write_text('framework:\n  name: QwenPI_v3\n  qwenvl:\n    base_vlm: Qwen3-VL-test\ntrainer: {}\n')
            (path / 'dataset_statistics.json').write_text(json.dumps({'test': {}}))
            def build(cfg):
                self.assertIs(loading.QWen3.Qwen3VLForConditionalGeneration, loading._ConfigOnlyQwen3)
                return SimpleNamespace()
            with patch.object(loading, 'apply_config_compat'), patch.object(loading, 'build_framework', side_effect=build):
                model = loading.from_config_only(path)
                self.assertEqual(model.norm_stats, {'test': {}})
            self.assertIs(loading.QWen3.Qwen3VLForConditionalGeneration, original)
            with patch.object(loading, 'apply_config_compat'), patch.object(loading, 'build_framework', side_effect=RuntimeError('construction failed')):
                with self.assertRaisesRegex(RuntimeError, 'construction failed'):
                    loading.from_config_only(path)
            self.assertIs(loading.QWen3.Qwen3VLForConditionalGeneration, original)


if __name__ == '__main__':
    unittest.main()
