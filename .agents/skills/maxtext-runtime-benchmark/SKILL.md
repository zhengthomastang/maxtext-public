---
name: maxtext-runtime-benchmark
description: Run and interpret paired MaxText runtime benchmarks for an existing model recipe. Use for latency, throughput, or memory comparisons that need a reproducible baseline and numerical checks.
---

# MaxText runtime benchmarks

Start with a native MaxText recipe or the [paired post-training experiments](../../../benchmarks/Post_Training_Experiments.md). Save the run settings, per-batch numerical checks, and per-step timing samples.

1. Pick one workload, one baseline, and one change. Match model and tokenizer revisions, batch order, valid token counts, precision, optimizer, compiler options, and device topology.
2. Verify numerical parity on the same input before interpreting speed. For SFT and DPO, compare loss and whole-model raw-gradient relative L2 at the **0.5%** gate. Check SFT z-loss and DPO policy and reference scores. Test SFT overflow separately when possible. For lossy changes, set a held-out quality budget before testing.
3. Compile and warm both paths on an authorized accelerator. Alternate synchronized steps when both paths fit together. Use fresh processes for cold-wall comparisons, with matching startup conditions and all required setup included.
4. Lead with the [standard MaxText metrics](../../../benchmarks/Getting_Started_Benchmarking.md#standard-maxtext-metrics). Report step time, tokens/s/device, model TFLOP/s/device, and estimated MFU with the precision, token count, FLOP convention, and hardware peak. Derive aggregate throughput from the total work and elapsed step time.
5. Save p10, median, p90, useful throughput, cold wall time, peak memory, and parity in the experiment report. Record separate input transfer, cache build, checkpoint reads, and compilation when they affect the job. Scope component results to the measured head, encoder, or attention block. For VLMs and world models, match media resolution, frame count, and downstream quality.

Use compiler defaults for the primary BF16 comparison. Use a separate matched deterministic run to diagnose numerical failures.
