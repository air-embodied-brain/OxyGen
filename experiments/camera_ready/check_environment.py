"""Run inside each isolated Python environment before any measurement."""
import importlib.metadata as md
import json
import platform
import sys
import torch
import transformers

name=sys.argv[1]
expected={
 'pi':{'torch':'2.6.0','transformers':'4.53.2','jax':'0.5.3','flax':'0.10.2'},
 'xiaomi':{'torch':'2.8.0','transformers':'4.57.1','flash-attn':'2.8.3'},
 'starvla':{'torch':'2.8.0','transformers':'4.57.0'},
 'qwen35':{'torch':'2.8.0','transformers':'5.2.0','flash-linear-attention':'0.5.1','causal-conv1d':'1.6.2.post1'},
}[name]
for package,version in expected.items():
 actual=md.version(package).split('+')[0]
 if actual!=version: raise RuntimeError(f'{name}: {package}={actual}, expected {version}')
if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
if name=='pi':
 import jax
 from openpi.policies.policy_config import create_trained_policy
 from experiments.common.setup import create_policy
 if not any(d.platform=='gpu' for d in jax.devices()): raise RuntimeError('JAX GPU unavailable')
 import matplotlib
elif name=='xiaomi':
 import flash_attn
 from oxygen_runtime import PrefixState, StaticLanguageCache
else:
 import oxygen_k_expert_formal_fixed_prefix
 if name=='starvla':
  from starVLA.model.framework.base_framework import _auto_import_framework_modules
  _auto_import_framework_modules()
 if name=='qwen35':
  import starVLA_qwen35_tp_benchmark
  from transformers.models.qwen3_5 import modeling_qwen3_5 as m
  if not m.is_fast_path_available: raise RuntimeError('Qwen3.5 fused fast path unavailable')
print(json.dumps({'python':platform.python_version(),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'packages':dict(sorted((d.metadata['Name'],d.version) for d in md.distributions() if d.metadata['Name']))}))
