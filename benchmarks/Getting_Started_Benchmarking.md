### Getting started with benchmark running in MaxText

Two approaches are here:

1. Run a model recipe with a single CLI command. Great to replicate performance results previously measured. See https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium for examples, and [`deepseek3_671b_tpu7x/`](deepseek3_671b_tpu7x/README.md) for DeepSeek-V3 671B FP8 recipes on TPU7x (Ironwood).
2. Run several experiments pythonically across a sweep of parameters (cluster configuration, maxtext parameters) with Cluster Toolkit workloads.

#### Paired SFT benchmark

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

Keep the generated NPZ for the Qwen3 comparisons. Select the matching checkpoint and `--model-name` for each row. Expose the exact number of H100 80 GB devices to JAX.

| Qwen3 size | Devices | Additional options |
| --- | ---: | --- |
| 0.6B, 1.7B | 1 | Defaults |
| 4B, 8B | 4 | `--tensor-parallel-size 4 --shard-mode auto` |
| [14B](https://huggingface.co/Qwen/Qwen3-14B/tree/40c069824f4251a91eefaf281ebe4c544efd3e18), [32B](https://huggingface.co/Qwen/Qwen3-32B/tree/9216db5781bf21249d130ec9da846c4624c16137) | 8 | `--tensor-parallel-size 8 --shard-mode auto --stream-checkpoint --sequential-arms` |

Sequential timing reloads the same checkpoint and fresh optimizer state for each arm. It excludes loading and compilation. For the eight-GPU setup, use `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`, `XLA_CLIENT_MEM_FRACTION=0.96`, and `XLA_PYTHON_CLIENT_PREALLOCATE=false` in both arms. The BFC allocator could not allocate the 32B tiled step's temporary buffers on the test node. [JAX GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)

Keep XLA compiler defaults for the primary comparison. Explicit sharding and `--compile-xla-flags=--xla_gpu_deterministic_ops=True` are separate comparisons. The runner applies either change to both arms.

#### Paired BF16 DPO benchmark

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

The first process uses native Tunix dense scores. The second keeps final normalization on the full sequence. It reads the completion mask from the final query row of Tunix's key-masked causal matrix and packs valid positions across chosen and rejected rows.

For a three-tile batch with a short final tile, both forward passes project the first two tiles together and use a smaller projection for the last tile. With the recipe above, the packed forward widths are 1,024 and 320 rows. Other batches keep the original 512-row forward tiles. Policy backward keeps 512-row score tiles and computes the target-logit cotangent with a pointwise select that XLA can fuse. It scatters logit cotangents back to their original positions, then applies the full-width projection pullback to preserve BF16 rounding.

The runner checks all eight matched batches at a **0.5%** raw-gradient relative L2 gate before timing. Both arms time the frozen reference, policy forward and backward, and AdamW update. The 64 steps cover eight timed passes through the batch cohort. The scorer is enabled in this benchmark. The standard MaxText DPO trainer uses native Tunix scoring.

For peak GPU memory, repeat each passing arm in a fresh process with `--profile-memory` and a separate output path. The profile reuses the verified cohort and runs the same updates without repeating the gradient capture. Sample physical GPU memory on the isolated devices between the `DPO_GPU_MEMORY_WINDOW_BEGIN` and `DPO_GPU_MEMORY_WINDOW_END` log markers. Keep the same allocator settings and report the peak per device. The JSON records allocator counters at each boundary and separate XLA temporary-memory estimates for policy and reference. Use the original runs for latency.

Add `--donate-policy` to let the native AdamW update reuse policy parameter buffers. The frozen reference gets independent device storage and continues to score online.

For the lower-budget BFC runs, use `--chunk-size 128 --donate-policy` and set `XLA_CLIENT_MEM_FRACTION=0.55` on Qwen3-1.7B or `0.60` on Qwen3-4B. Keep preallocation disabled and use the same empty XLA compiler flags file. Run native dense at the default BFC budget as the deployment baseline and at the candidate budget as a capacity control. Record each arm's budget with its latency and sampled memory.

For new GKE runs, submit each benchmark as a Cluster Toolkit JobSet. Authenticate
to the cluster and configure `gcluster` before submitting:

```bash
export PROJECT_ID=<PROJECT_ID>
export GKE_CLUSTER=<CLUSTER_NAME>
export LOCATION=<LOCATION> # e.g. europe-west4 (region) or us-central1-a (zone)
export RUN_NAME=<RUN_NAME>
export DOCKER_IMAGE="us-docker.pkg.dev/cloud-tpu-images/maxtext-images/tpu_pre_training:latest"
export COMPUTE_TYPE=<COMPUTE_TYPE>
export TOPOLOGY=<TOPOLOGY>
export BASE_OUTPUT_DIRECTORY=<GCS_BUCKET> # e.g. gs://my-bucket/maxtext-runs

gcloud config set project ${PROJECT_ID?}
gcloud container clusters get-credentials ${GKE_CLUSTER?} \
  --location ${LOCATION?} \
  --project ${PROJECT_ID?}
gcluster job config set project ${PROJECT_ID?}
gcluster job config set cluster ${GKE_CLUSTER?}
gcluster job config set location ${LOCATION?}
```

Then submit the benchmark workload for the target cluster:

```bash
# Cluster Toolkit: McJAX
gcluster job submit \
  --image=${DOCKER_IMAGE?} \
  --command="python3 -m benchmarks.benchmark_runner on-device --base_output_directory=${BASE_OUTPUT_DIRECTORY?} --run_name=${RUN_NAME?} --num_steps=5" \
  --name=${RUN_NAME?} \
  --compute-type=${COMPUTE_TYPE?} \
  --topology=${TOPOLOGY?}

# Cluster Toolkit: Pathways
gcluster job submit \
  --image=${DOCKER_IMAGE?} \
  --name=${RUN_NAME?} \
  --pathways \
  --compute-type=${COMPUTE_TYPE?} \
  --topology=${TOPOLOGY?} \
  --num-slices=1 \
  --pathways-gcs-location=${BASE_OUTPUT_DIRECTORY?} \
  --command="python3 -m benchmarks.benchmark_runner on-device --use_pathways=true --base_output_directory=${BASE_OUTPUT_DIRECTORY?} --run_name=${RUN_NAME?} --num_steps=5"
```

- **xla_flags_library.py**: A grouping of xla flags organized by purpose with details on how they can be applied to a model.
- **maxtext_trillium_model_configs.py**: A list of model definitions for Trillium. See optimized models here and how they apply xla flags. This config provides a pythonic way to run MaxText models.
- **benchmark_runner.py**: A cli interface to running a specific model recipe, on pathways or mcjax directly or with orchestration like Cluster Toolkit (`ctk`) with one command.

```shell
# McJax with Cluster Toolkit (ctk)
CLUSTER=my-cluster
LOCATION=my-location
PROJECT=my-project
python3 -m benchmarks.benchmark_runner ctk --project ${PROJECT?} --location ${LOCATION?} --cluster_name ${CLUSTER?} --device_type v6e-256 --base_output_directory gs://maxtext-experiments-tpem/ --num_steps=5
```

```shell
# Pathways with Cluster Toolkit (ctk)
export RUNNER=us-docker.pkg.dev/path/to/maxtext_runner
export PROXY_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images/cloud-pathways/proxy_server
export SERVER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images/cloud-pathways/server

python3 -m benchmarks.benchmark_runner ctk --use_pathways=true --pathways_server_image="${SERVER_IMAGE?}" --pathways_proxy_server_image="${PROXY_IMAGE?}" --pathways_runner_image="${RUNNER?}" --project ${PROJECT?} --location ${LOCATION?} --cluster_name ${CLUSTER?} --device_type v6e-256 --base_output_directory gs://maxtext-experiments-tpem/ --num_steps=5
```

```shell
# On-device
# Run model benchmark on current device (must run same command on all workers).
python3 -m benchmarks.benchmark_runner on-device --base_output_directory gs://maxtext-experiments-tpem/ --run_name="test-run" --num_steps=5
```

- **maxtext_ctk_runner.py** (and backward-compatible alias **maxtext_xpk_runner.py**): Python orchestration for Cluster Toolkit (`gcluster`) workloads.

```shell
# Loop possibilities:
  # 1. Test different libtpu nightly versions.
  #  for libtpu_type in [
  #           LibTpuType.NIGHTLY
  #       ]:
  #     todays_date = time.strftime('%Y%m%d')
  #    for date in ['20241201', '20241202', todays_date]:

  # 2. Test different model configurations.
  # for remat_policy in ['qkv_proj_offloaded', 'minimal']:
  #   model.tuning_params['remat_policy'] = remat_policy

for model in list_of_models:
    # Run workloads on the below clusters
    for cluster_config in [
      v5e_cluster_config,
      v6e_cluster_config,
    ]:
      # Run workloads in the following slice configurations
      for num_slices in [1, 2 , 4]:
        # Use the libtpu dependencies from:
        for libtpu_type in [
            LibTpuType.MAXTEXT
        ]:
          wl_config = WorkloadConfig(
            model=model,
            num_slices=num_slices,
            device_type=cluster_config.device_type,
            base_output_directory=base_output_dir,
            priority="medium",
            max_restarts=0,
            libtpu_type=libtpu_type,
            libtpu_nightly_version="",
            base_docker_image=base_docker_image,
            pathways_config=None
          )
          command, name = generate_workload_cmd(
            cluster_config=cluster_config,
            wl_config=wl_config
          )

          print(f"Name of the workload is: {name} \n")
          workload_names.append(name)

          print(f"gcluster command to be used is: {command} \n")
          workload_cmds.append(command)

  for workload_name, workload_cmd in zip(workload_names, workload_cmds):
    return_code = run_command_with_updates(workload_cmd, workload_name)
    if return_code != 0:
      print(f'Unable to run gcluster workload: {workload_name}')

```
