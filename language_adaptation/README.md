# Language adaptation

This directory contains language-training and serving features, separate from
the system-performance benchmarks in [`experiments`](../experiments/).

The current feature is the [LIBERO textual-memory adaptation](libero/README.md),
which trains a suffix-only LoRA to generate observation-grounded textual memory
alongside the frozen action path.
