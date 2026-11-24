# Post-training benchmark experiments

Use the [standard MaxText metrics](Getting_Started_Benchmarking.md#standard-maxtext-metrics) for the main comparison. Save the raw report with the model and dataset revisions, device count, and allocator and compiler settings.

- [QASPER supervised fine-tuning](#paired-sft-benchmark)
- [ORCA preference optimization](#paired-bf16-dpo-benchmark)
- [Frozen-reference cache](#precompute-a-fixed-reference)

Keep p10, p50, p90, peak device memory, and cold wall time with the experiment. Report useful throughput as SFT nonpadding or target tokens/s/device, and DPO preference pairs or completion tokens/s/device. Divide the total useful work by the summed step duration and device count. A parity result applies to the matched inputs and shared weights checked.

For these standalone runners, derive the standard metrics from the raw step durations and label them as estimates where applicable. For the standard configured-token rate in DPO, count preference pairs × padded sequence length per step and divide by elapsed step time and device count. If reporting decoder slots instead, say so explicitly. Tunix scores chosen and rejected sequences as two decoder rows per pair. The DPO model FLOP estimate counts the policy forward and backward on those rows, plus one frozen-reference forward when it runs every step. Account for one-time reference scoring separately.

## Paired SFT benchmark

### Run the QASPER workload

The runner checks loss, z-loss, and raw-gradient parity before timing synchronized full AdamW steps. It replays the same tokenized batches, optimizer settings, and device topology within each pair. The JSON report captures batch hashes, batch order, compiler flags, per-batch checks, and p10, median, and p90 latency.

Download the public [QASPER archive](https://allenai.org/data/qasper) and use a local snapshot of [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e).

```bash
python -m benchmarks.prepare_qasper_sft_batches \
  --archive "$QASPER_ARCHIVE" --tokenizer-path "$QWEN_CHECKPOINT" \
  --tokenizer-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --output /tmp/qasper-s4096.npz

python -m benchmarks.supervised_sft_projection \
  --model-name qwen3-1.7b --dtype bfloat16 --max-target-length 4096 \
  --checkpoint-layout safetensors_dynamic \
  --load-parameters-path "$QWEN_CHECKPOINT" --tokenizer-path "$QWEN_CHECKPOINT" \
  --batches /tmp/qasper-s4096.npz --output-directory /tmp/maxtext-sft-results \
  --scan-layers true --warmup 4 --repeats 32
```

### Change model size or sharding

Keep the generated NPZ for the Qwen3 comparisons. Select the matching checkpoint and `--model-name` for each row. Expose the exact number of H100 80 GB devices to JAX.

| Qwen3 size | Devices | Additional options |
| --- | ---: | --- |
| 0.6B, 1.7B | 1 | Defaults |
| 4B, 8B | 4 | `--tensor-parallel-size 4 --shard-mode auto` |
| [14B](https://huggingface.co/Qwen/Qwen3-14B/tree/40c069824f4251a91eefaf281ebe4c544efd3e18), [32B](https://huggingface.co/Qwen/Qwen3-32B/tree/9216db5781bf21249d130ec9da846c4624c16137) | 8 | `--tensor-parallel-size 8 --shard-mode auto --stream-checkpoint --sequential-arms` |

Sequential timing reloads the same checkpoint and fresh optimizer state for each arm. It excludes loading and compilation. For the eight-GPU setup, use `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`, `XLA_CLIENT_MEM_FRACTION=0.96`, and `XLA_PYTHON_CLIENT_PREALLOCATE=false` in both arms. The BFC allocator could not allocate the 32B tiled step's temporary buffers on the test node. [JAX GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)

Keep XLA compiler defaults for the primary comparison. Explicit sharding and `--compile-xla-flags=--xla_gpu_deterministic_ops=True` are separate comparisons. The runner applies either change to both arms.

## Paired BF16 DPO benchmark

### Run the ORCA workload

Use a MaxText post-training environment with [Tunix at the tested revision](https://github.com/google/tunix/commit/21fa70f92aadd966cea68efcb73c9786710b8c46). Set `QWEN_CHECKPOINT` to a local [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca), [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e), or [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B/tree/1cfa9a7208912126459214e8b04321603b3df60c) snapshot. Set `ORCA_PARQUET` to the pinned [Argilla ORCA training Parquet directory](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs/tree/0b10ec0df32c919f95126b203c8f5962b6875896). Use a distinct `DPO_BENCHMARK_DIR` per model and device mesh, with room for raw gradients (20 GiB for 0.6B and 128 GiB for 4B). Expose exactly `TP_SIZE` devices, and keep the same mesh in both arms. An empty XLA flags file keeps compiler defaults and prevents MaxText from adding HLO dump flags.

```bash
mkdir -p "$DPO_BENCHMARK_DIR"
: > "$DPO_BENCHMARK_DIR/xla-default.flags"
export XLA_FLAGS="$DPO_BENCHMARK_DIR/xla-default.flags"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset XLA_PYTHON_CLIENT_ALLOCATOR XLA_CLIENT_MEM_FRACTION XLA_PYTHON_CLIENT_MEM_FRACTION

MODEL_NAME=qwen3-0.6b
TP_SIZE=1
ORCA_COHORT_SHA=45711c62dd9b6a98ab1bfb74eb9beacc479f7fe90e6cc2665948ed2823c1dff3

for chunk_size in 0 128; do
  python -m benchmarks.dpo_completion_scoring \
    --model-name "$MODEL_NAME" --tensor-parallel-size "$TP_SIZE" \
    --checkpoint "$QWEN_CHECKPOINT" --preference-directory "$ORCA_PARQUET" \
    --cohort-sha256 "$ORCA_COHORT_SHA" \
    --parity-directory "$DPO_BENCHMARK_DIR/reference" \
    --output "$DPO_BENCHMARK_DIR/chunk-$chunk_size.json" \
    --chunk-size "$chunk_size" --warmup 4 --repeats 64 || exit
done
```

For Qwen3-1.7B, set `MODEL_NAME=qwen3-1.7b` and `TP_SIZE=1`. For Qwen3-4B, set `MODEL_NAME=qwen3-4b` and `TP_SIZE=2` for automatic tensor sharding on two H100s. Add `--stream-checkpoint` in both arms for these larger checkpoints.

### Validate the scorer and measure the full step

The first process uses native Tunix dense scores. The second keeps final normalization on the full sequence. It reads the completion mask from the final query row of Tunix's key-masked causal matrix and packs valid positions across chosen and rejected rows.

For a three-tile batch with a short final tile, both forward passes project the first two tiles together and use a smaller projection for the last tile. With the recipe above, the packed forward widths are 1,024 and 320 rows. Other batches keep the original 512-row forward tiles. Policy backward keeps 512-row score tiles and computes the target-logit cotangent with a pointwise select that XLA can fuse. It scatters logit cotangents back to their original positions, then applies the full-width projection pullback to preserve BF16 rounding.

The runner checks all eight matched batches at a **0.5%** raw-gradient relative L2 gate before timing. Both arms time the frozen reference, policy forward and backward, and AdamW update. The 64 steps cover eight timed passes through the batch cohort. The scorer is enabled in this benchmark. The standard MaxText DPO trainer uses native Tunix scoring.

### Profile peak GPU memory

For peak GPU memory, repeat each passing arm in a fresh process with `--profile-memory` and a separate output path. The profile reuses the verified cohort and runs the same updates without repeating the gradient capture. Sample physical GPU memory on the isolated devices between the `DPO_GPU_MEMORY_WINDOW_BEGIN` and `DPO_GPU_MEMORY_WINDOW_END` log markers. Keep the same allocator settings and report the peak per device. The JSON records allocator counters at each boundary and separate XLA temporary-memory estimates for policy and reference. Use the original runs for latency.

### Test policy-buffer donation

Add `--donate-policy` to let the native AdamW update reuse policy parameter buffers. The frozen reference gets independent device storage and continues to score online.

For the lower-budget BFC runs, use `--chunk-size 0 --donate-policy` at `XLA_CLIENT_MEM_FRACTION=0.55` on Qwen3-1.7B, and `--chunk-size 128 --donate-policy` at `0.60` on Qwen3-4B. Keep preallocation disabled and use the same empty XLA compiler flags file. Run native dense with stock donation at the default BFC budget as the deployment baseline and at the candidate budget as a capacity control. Save gradients from the stock native scorer before running either donated arm. Record each arm's budget with its latency and sampled memory.

### Precompute a fixed reference

For a fixed checkpoint and tokenized cohort, add `--reference-cache "$DPO_BENCHMARK_DIR/reference-scores.json" --build-reference-cache` to the model and data flags with `--chunk-size 0`. This runs native reference scoring and exits. Give the builder a separate `--output`. In a new process, pass the same `--reference-cache` with `--chunk-size 0 --donate-policy` and omit `--build-reference-cache`. The key checks the checkpoint shards, model and scoring code, tokenized cohort, mesh, and compiler options.

Use the same BFC budget for the builder, cached training, and online dense control. For the Qwen3 GPU comparison, use 55% on 1.7B and 60% on 4B, with preallocation disabled and an empty XLA flags file. Sample memory over both cache phases and compare the higher peak with the online whole-process peak.

### Measure cold wall time

Check the **0.5%** gradient gate separately, then validate the timed cohort against the parity run. Use fresh processes with `JAX_ENABLE_COMPILATION_CACHE=false`. Add `--skip-compiled-memory-analysis` to both training arms so the online process does no extra reference compilation for diagnostics. Include imports, checkpoint load, scoring, compilation, warm-ups, and measured steps. For the on-disk cache, sum builder and training wall times. Reverse the order and report steady p90 separately. Sample whole-process peak memory in independent runs.

For BF16 runs, repeat the gradient gate in independent processes. Pass `--record-reference-scores` to both timing and memory runs. The `reference_scores_by_batch` field records the first chosen and rejected score vectors used for each batch. The runner copies them to the host after the update loop. Compare the measured score vectors with the paired dense values and record any mismatch as a failed comparison. Also require a separate raw-gradient check.
