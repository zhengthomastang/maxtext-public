# Benchmarking with MaxText

- **Run one recipe** to reproduce a known workload. See the [Trillium recipes](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium) or [DeepSeek-V3 on Ironwood](deepseek3_671b_tpu7x/README.md).
- **Sweep model or cluster settings** with the [Cluster Toolkit recipes](#cluster-toolkit-benchmarks).
- **Compare two post-training paths** with the [SFT and DPO experiments](Post_Training_Experiments.md).

## Standard MaxText metrics

Report the baseline and candidate on the same workload after warm-up. Keep the model, batch shape, precision, device mesh, and compiler settings fixed.

| Metric | Native key or calculation |
| --- | --- |
| Step time | `perf/step_time_seconds` |
| Tokens/s/device | `perf/per_device_tokens_per_sec`. Uses configured sequence positions, including padding. |
| Model TFLOP/s/device | `perf/per_device_tflops_per_sec`. Divides theoretical model TFLOPs per device per step by step time. |
| Model FLOPs utilization (MFU) | Model TFLOP/s/device divided by the device's peak TFLOP/s for the reported precision. |

Use mean synchronized step time to derive aggregate throughput from standalone runners. State the token and FLOP counting conventions and the hardware peak. MFU comparisons are easiest to interpret when theoretical model work is fixed. When an optimization removes work, pair MFU with step time and token throughput. [MaxText metric logs](../docs/guides/monitoring_and_debugging/understand_logs_and_metrics.md) · [MFU definition](../docs/reference/performance_metrics.md)

Keep raw timing, tail latency, useful work, correctness, peak memory, and startup costs in the [experiment report](Post_Training_Experiments.md).

## Paired SFT benchmark

[Run the QASPER replay](Post_Training_Experiments.md#paired-sft-benchmark) to compare native vocabulary tiling with supervised-position projection.

## Paired BF16 DPO benchmark

[Run the ORCA replay](Post_Training_Experiments.md#paired-bf16-dpo-benchmark) to compare dense with optimized DPO under matched compiler and allocator settings.

### Precompute a fixed reference

[Measure frozen-reference reuse](Post_Training_Experiments.md#precompute-a-fixed-reference), including the cache-build cost in cold wall time and peak memory.

## Cluster Toolkit benchmarks

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
