---
name: maxtext-runtime-benchmark
description: Run and interpret paired MaxText runtime benchmarks for an existing model recipe. Use for latency, throughput, or memory comparisons that need a reproducible baseline and numerical checks.
---

# MaxText runtime benchmarks

Use the [paired SFT](../../../benchmarks/Getting_Started_Benchmarking.md#paired-sft-benchmark) or [paired BF16 DPO](../../../benchmarks/Getting_Started_Benchmarking.md#paired-bf16-dpo-benchmark) runner for these workloads. Both emit per-batch checks and per-step timings as JSON. For other workloads, start from the corresponding native recipe.

1. Pick one workload, one baseline, and one proposed change. Match the model and tokenizer revisions, batch order, valid token counts, weight and activation precision, optimizer, compiler options, and device topology. Record the source revision and environment.
2. Use an available accelerator in the user-authorized environment. Compile and warm both paths first. Alternate synchronized baseline and candidate steps and report the median, p10, p90, and useful tokens per second. Record input transfer, cache build, cold reads, and compilation separately when they affect end-to-end time.
3. Verify numerical parity on the same input before interpreting speed. For SFT and DPO, compare losses and whole-model raw-gradient relative L2 at the 0.5% gate. Check SFT z-loss and DPO reference and policy scores. Treat SFT overflow as a separate workload when possible. For lossy compression or quantization, use a predetermined held-out quality budget instead.
4. Report baseline and candidate values within each matched workload. Treat head, encoder, and attention probes as component evidence. For VLM or world-model work, keep media resolution, frames, and the downstream quality metric fixed when comparing runtimes.

Keep compiler defaults for the primary BF16 comparison. A separate matched
deterministic run can help diagnose a failure. Timing conclusions apply to the
compiler settings that passed parity.
