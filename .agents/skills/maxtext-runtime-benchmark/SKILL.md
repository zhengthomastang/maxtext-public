---
name: maxtext-runtime-benchmark
description: Run and interpret paired MaxText runtime benchmarks for an existing model recipe. Use for latency, throughput, or memory comparisons that need a reproducible baseline and numerical checks.
---

# MaxText runtime benchmarks

Start from the [benchmark guide](../../../benchmarks/Getting_Started_Benchmarking.md#paired-sft-benchmark). The paired SFT runner is an executable entry point with a JSON report. Use the existing recipe runner for other native MaxText training jobs.

1. Pick one workload, one baseline, and one proposed change. Match the model and tokenizer revisions, batch order, valid token counts, weight and activation precision, optimizer, and device topology. Record the source revision and environment.
2. Use an available accelerator in the user-authorized environment. Compile and warm both paths first. Alternate synchronized baseline and candidate steps and report the median, p10, p90, and useful tokens per second. Record input transfer, cache build, cold reads, and compilation separately when they affect end-to-end time.
3. Verify numerical parity on the same input before interpreting speed. For supervised projection, compare loss, z-loss, and whole-model raw gradient relative L2 using the default 0.5% gate. Treat overflow as a separate workload when possible. For lossy compression or quantization, use a predetermined held-out quality budget instead.
4. Report baseline and candidate values within each matched workload. Treat head, encoder, and attention probes as component evidence. For VLM or world-model work, keep media resolution, frames, and the downstream quality metric fixed when comparing runtimes.

Report the scope of the result and keep the per-step JSON so another agent can check or reproduce it.
