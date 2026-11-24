# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paired full MaxText SFT steps from one model checkpoint and a fixed batch."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.trainers.pre_train import train
from maxtext.utils import sharding, train_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR
import numpy as np
from transformers import AutoTokenizer


def parse_args():
  """Parse and validate the paired benchmark options."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--load-parameters-path", required=True)
  parser.add_argument("--checkpoint-layout", choices=["orbax", "safetensors_dynamic"], default="orbax")
  parser.add_argument("--tokenizer-path", required=True)
  parser.add_argument("--model-name", default="qwen3-0.6b")
  parser.add_argument("--output-directory", required=True, type=Path)
  parser.add_argument("--max-target-length", default=2048, type=int)
  parser.add_argument("--batch-size", default=1, type=int)
  parser.add_argument("--completion-tokens", default=240, type=int)
  parser.add_argument("--batches", type=Path, help="NPZ of preprocessed MaxText SFT batches.")
  parser.add_argument(
      "--parity-batches", type=int, help="Limit parity batches for smoke runs; default checks all batches."
  )
  parser.add_argument("--num-vocab-tiling", default=8, type=int)
  parser.add_argument("--projection-tiles", default=1, type=int)
  parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
  parser.add_argument("--attention", default="dot_product")
  parser.add_argument("--matmul-precision", choices=["default", "high", "highest"], default="highest")
  parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
  parser.add_argument("--z-loss-multiplier", default=1e-4, type=float)
  parser.add_argument("--scan-layers", choices=["true", "false"])
  parser.add_argument("--warmup", default=3, type=int)
  parser.add_argument("--repeats", default=12, type=int)
  parser.add_argument("--loss-rtol", default=1e-5, type=float)
  parser.add_argument("--gradient-relative-l2-tolerance", default=5e-3, type=float)
  args = parser.parse_args()
  seq, batch = args.max_target_length, args.batch_size
  tile_count = args.num_vocab_tiling
  if batch < 1 or seq < 2 or (args.parity_batches is not None and args.parity_batches < 1):
    parser.error("Require positive batch counts and a sequence length of at least two.")
  if tile_count < 2 or (batch * seq) % tile_count:
    parser.error("Use at least two tiles that divide the number of batch positions.")
  if not 0 < args.projection_tiles < tile_count:
    parser.error("Projection tiles must be fewer than the baseline tile count.")
  if args.batches is None and not 0 < args.completion_tokens < seq:
    parser.error("The fixed fixture needs a nonempty prompt and completion.")
  if args.warmup < 1 or args.repeats < 2:
    parser.error("Use at least one warm-up and two timed repetitions.")
  return args


def make_config(args, name, projection_tiles):
  """Build one native SFT configuration with the chosen output-head path."""
  overrides = {
      "run_name": name,
      "base_output_directory": str(args.output_directory.resolve()),
      "load_parameters_path": args.load_parameters_path,
      "source_checkpoint_layout": args.checkpoint_layout,
      "model_name": args.model_name,
      "hardware": "gpu",
      "attention": args.attention,
      "dtype": args.dtype,
      "weight_dtype": "float32",
      "grad_dtype": "float32",
      "logits_dot_in_fp32": "true",
      "matmul_precision": args.matmul_precision,
      "z_loss_multiplier": args.z_loss_multiplier,
      "max_target_length": args.max_target_length,
      "max_prefill_predict_length": min(128, args.max_target_length // 2),
      "per_device_batch_size": args.batch_size,
      "num_vocab_tiling": args.num_vocab_tiling,
      "sft_supervised_projection_tiles": projection_tiles,
      "opt_type": args.optimizer,
      "learning_rate": 2e-5,
      "warmup_steps_fraction": 0.0,
      "steps": args.warmup + args.repeats + 4,
      "gradient_accumulation_steps": 1,
      "enable_dropout": "false",
      "enable_checkpointing": "false",
      "enable_tensorboard": "false",
      "enable_goodput_recording": "false",
      "record_internal_nn_metrics": "false",
      "skip_jax_distributed_system": "true",
      "use_sft": "true",
      "sft_train_on_completion_only": "true",
      "use_tunix_gradient_accumulation": "false",
      "dataset_type": "synthetic",
      "packing": "false",
      "tokenizer_type": "huggingface",
      "tokenizer_path": args.tokenizer_path,
      "log_config": "false",
  }
  if args.scan_layers is not None:
    overrides["scan_layers"] = args.scan_layers
  path = Path(MAXTEXT_PKG_DIR) / "configs" / "post_train" / "sft.yml"
  return pyconfig.initialize([sys.argv[0], str(path), *(f"{key}={value}" for key, value in overrides.items())])


def fixed_batch(args, vocab_size):
  """Build a repeatable prompt and response fixture for smoke runs."""
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
  prompt = (
      "User: Explain how a language model uses the tokens in a long prompt to answer a short question. "
      "Discuss causal attention, masked training labels, and the output projection. "
  )
  completion = (
      "Assistant: Causal attention preserves preceding tokens as context. The training objective applies "
      "only to the response labels. The decoder still consumes the entire input sequence. "
  )

  def tokens_exact(text, length):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
      raise ValueError("The tokenizer produced an empty sequence.")
    return (ids * ((length + len(ids) - 1) // len(ids)))[:length]

  length, completion_count = args.max_target_length, args.completion_tokens
  prompt_count = length + 1 - completion_count
  rows = [
      tokens_exact(f"Participant {row}. " + prompt, prompt_count)
      + tokens_exact(completion, completion_count - 1)
      + [tokenizer.eos_token_id]
      for row in range(args.batch_size)
  ]
  tokens = np.asarray(rows, dtype=np.int32)
  if np.min(tokens) < 0 or np.max(tokens) >= vocab_size:
    raise ValueError("Fixture token IDs are incompatible with the MaxText vocabulary.")
  positions = np.broadcast_to(np.arange(length, dtype=np.int32), (args.batch_size, length)).copy()
  target_mask = (positions >= prompt_count - 1).astype(np.int32)
  return {
      "inputs": tokens[:, :-1],
      "targets": tokens[:, 1:],
      "inputs_position": positions,
      "inputs_segmentation": np.ones_like(positions),
      "targets_segmentation": target_mask,
  }


def load_batches(args, vocab_size):
  """Load a cohort and record the actual sequence and supervision counts."""
  if args.batches is None:
    batches = [fixed_batch(args, vocab_size)]
    source = {"kind": "fixed_fixture"}
  else:
    fields = ("inputs", "targets", "inputs_position", "inputs_segmentation", "targets_segmentation")
    with args.batches.open("rb") as handle:
      source = {"kind": "npz", "sha256": hashlib.file_digest(handle, "sha256").hexdigest()}
    with np.load(args.batches, allow_pickle=False) as archive:
      arrays = {field: archive[field] for field in fields}
    shape = arrays["inputs"].shape
    if (
        len(shape) != 3
        or shape[0] < 1
        or shape[1:] != (args.batch_size, args.max_target_length)
        or any(x.shape != shape or x.dtype != np.int32 for x in arrays.values())
    ):
      raise ValueError("Each NPZ field must be an int32 array of shape [num_batches, batch_size, sequence_length].")
    batches = [{field: values[i] for field, values in arrays.items()} for i in range(shape[0])]

  capacity = args.projection_tiles * (args.batch_size * args.max_target_length // args.num_vocab_tiling)
  useful_tokens, supervised_tokens, occupied_tiles = [], [], []
  for batch in batches:
    for field in ("inputs", "targets"):
      if np.min(batch[field]) < 0 or np.max(batch[field]) >= vocab_size:
        raise ValueError(f"{field} contains token IDs outside the model vocabulary.")
    count = int(np.count_nonzero(batch["targets_segmentation"]))
    if not 0 < count <= capacity:
      raise ValueError(f"Batch has {count} supervised targets, outside the compact capacity of {capacity}.")
    useful_tokens.append(int(np.count_nonzero(batch["inputs_segmentation"])))
    supervised_tokens.append(count)
    mask_tiles = batch["targets_segmentation"].reshape(args.num_vocab_tiling, -1)
    occupied_tiles.append(int(np.count_nonzero(np.any(mask_tiles, axis=1))))
  source.update(
      {
          "input_tokens_by_batch": useful_tokens,
          "supervised_tokens_by_batch": supervised_tokens,
          "occupied_original_tiles_by_batch": occupied_tiles,
      }
  )
  return batches, source


def make_loss_and_gradient(model, config):
  graph, params, rest = nnx.split(model, nnx.Param, ...)

  def objective(weights, non_params, batch):
    local_model = nnx.merge(graph, weights, non_params, copy=True)
    loss, aux = train.loss_fn(local_model, config, dict(batch), None, None, is_train=True)
    return loss, (aux["z_loss"], aux["total_weights"])

  evaluate = jax.jit(jax.value_and_grad(objective, has_aux=True))
  return lambda batch: jax.block_until_ready(evaluate(params, rest, batch))


@jax.jit
def gradient_stats(candidate, baseline):
  """Return squared difference, squared reference norm, and max error per leaf."""

  def leaf_stats(actual, expected):
    actual = actual.astype(jnp.float32)
    expected = expected.astype(jnp.float32)
    diff = actual - expected
    return jnp.stack([jnp.sum(jnp.square(diff)), jnp.sum(jnp.square(expected)), jnp.max(jnp.abs(diff))])

  actual_leaves, candidate_layout = jax.tree.flatten(candidate)
  baseline_leaves, baseline_layout = jax.tree.flatten(baseline)
  if candidate_layout != baseline_layout:
    raise ValueError("Gradient trees differ.")
  stats = jnp.stack([leaf_stats(a, b) for a, b in zip(actual_leaves, baseline_leaves, strict=True)])
  return stats


def full_step(config, graph, state, shardings, mesh, batch):
  params_shardings, train_shardings = sharding.maybe_update_params_sharding_with_opt(config, shardings)
  data_sharding = sharding.get_input_data_sharding(config, mesh)
  jitted = train_utils.jit_train_step(
      config, graph, state, train_shardings, data_sharding, train.train_step, params_shardings, mesh
  )
  start = perf_counter()
  executable = jitted.lower(state, batch).compile()
  return executable, (perf_counter() - start) * 1000


def summarize(samples):
  p10, median, p90 = np.percentile(samples, [10, 50, 90])
  return {"p10_ms": float(p10), "median_ms": float(median), "p90_ms": float(p90), "samples_ms": samples}


def check_parity(args, expected_count, baseline, candidate):
  """Check both losses and collect raw gradient differences for one batch."""
  (loss_baseline, (z_baseline, count_baseline)), grad_baseline = baseline
  (loss_candidate, (z_candidate, count_candidate)), grad_candidate = candidate
  np.testing.assert_allclose(loss_candidate, loss_baseline, rtol=args.loss_rtol, atol=1e-6)
  np.testing.assert_allclose(z_candidate, z_baseline, rtol=args.loss_rtol, atol=1e-6)
  if int(count_candidate) != expected_count or int(count_baseline) != expected_count:
    raise AssertionError("Supervised target counts differ from the batch mask.")
  stats = np.asarray(gradient_stats(grad_candidate, grad_baseline))
  diff_sq, baseline_sq = np.sum(stats[:, :2], axis=0, dtype=np.float64)
  if baseline_sq <= 0:
    raise AssertionError("Reference gradient norm is zero.")
  paths = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(grad_baseline)]
  largest_errors = [
      {"parameter": paths[i], "difference_l2": float(stats[i, 0] ** 0.5), "baseline_l2": float(stats[i, 1] ** 0.5)}
      for i in np.argsort(stats[:, 0])[-3:][::-1]
  ]
  parity = {
      "loss_tiled": float(loss_baseline),
      "loss_supervised": float(loss_candidate),
      "z_loss_tiled": float(z_baseline),
      "z_loss_supervised": float(z_candidate),
      "supervised_targets": expected_count,
      "gradient_relative_l2_error": float((diff_sq / baseline_sq) ** 0.5),
      "gradient_max_abs_error": float(np.max(stats[:, 2])),
      "largest_gradient_differences": largest_errors,
      "loss_rtol": args.loss_rtol,
      "gradient_relative_l2_tolerance": args.gradient_relative_l2_tolerance,
  }
  print(json.dumps(parity, indent=2, allow_nan=False), flush=True)
  return parity


def main():
  args = parse_args()
  if jax.default_backend() != "gpu" or jax.device_count() != 1:
    raise RuntimeError("Expose exactly one GPU to JAX for this benchmark.")
  args.output_directory.mkdir(parents=True, exist_ok=True)
  configs = {
      "tiled": make_config(args, "tiled", 0),
      "supervised": make_config(args, "supervised", args.projection_tiles),
  }
  _, manager, shardings, model, mesh, *_, initial = train_utils.setup_train_loop(configs["tiled"], recorder=None)
  try:
    baseline_config = configs["tiled"]
    host_batches, source = load_batches(args, baseline_config.vocab_size)
    data_sharding = sharding.get_input_data_sharding(baseline_config, mesh)
    batches = jax.block_until_ready([jax.device_put(batch, data_sharding) for batch in host_batches])
    parity_fns = {name: make_loss_and_gradient(model, config) for name, config in configs.items()}
    parity = []
    for index, batch in enumerate(batches[: args.parity_batches]):
      results = {}
      for name, config in configs.items():
        with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules):
          results[name] = parity_fns[name](batch)
      checked = check_parity(args, source["supervised_tokens_by_batch"][index], results["tiled"], results["supervised"])
      del results
      parity.append(checked)
    worst_gradient_error = max(result["gradient_relative_l2_error"] for result in parity)
    if not worst_gradient_error <= args.gradient_relative_l2_tolerance:
      failure = {"parity": parity, "worst_gradient_relative_l2_error": worst_gradient_error}
      (args.output_directory / "parity_failure.json").write_text(
          json.dumps(failure, indent=2, allow_nan=False) + "\n", encoding="utf-8"
      )
      raise AssertionError(f"Worst gradient relative L2 error {worst_gradient_error:.6g} exceeds tolerance.")
    del parity_fns, model

    graph, original = nnx.split(initial)
    del initial
    candidate_state = jax.block_until_ready(jax.tree.map(lambda x: x.copy() if isinstance(x, jax.Array) else x, original))
    states = {"tiled": original, "supervised": candidate_state}
    del original, candidate_state
    executables, compile_times = {}, {}
    for name, config in configs.items():
      with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules):
        executables[name], compile_times[name] = full_step(config, graph, states[name], shardings, mesh, batches[0])

    samples = {name: [] for name in configs}
    timed_batch_indices = []
    for step in range(args.warmup + args.repeats):
      batch_index = step % len(batches)
      batch = batches[batch_index]
      if step >= args.warmup:
        timed_batch_indices.append(batch_index)
      names = tuple(configs) if step % 2 == 0 else tuple(reversed(configs))
      for name in names:
        start = perf_counter()
        states[name], _ = jax.block_until_ready(executables[name](states[name], batch))
        if step >= args.warmup:
          samples[name].append((perf_counter() - start) * 1000)
    timed_tokens = sum(source["input_tokens_by_batch"][index] for index in timed_batch_indices)
    full_step_metrics = {
        name: {**summarize(values), "input_tokens_per_second": 1000 * timed_tokens / sum(values)}
        for name, values in samples.items()
    }
    report = {
        "schema_version": 1,
        "scope": "native_sft_step",
        "device": jax.devices()[0].device_kind,
        "device_count": jax.device_count(),
        "jax_version": jax.__version__,
        "model": args.model_name,
        "checkpoint": args.load_parameters_path,
        "checkpoint_layout": args.checkpoint_layout,
        "dtype": baseline_config.dtype.name.lower(),
        "weight_dtype": "float32",
        "grad_dtype": "float32",
        "z_loss_multiplier": args.z_loss_multiplier,
        "attention": args.attention,
        "matmul_precision": args.matmul_precision,
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "sequence_length": args.max_target_length,
        "num_vocab_tiling": args.num_vocab_tiling,
        "projection_tiles": args.projection_tiles,
        "warmup": args.warmup,
        "input": source,
        "parity": parity,
        "worst_gradient_relative_l2_error": worst_gradient_error,
        "compile_ms": compile_times,
        "timed_batch_indices": timed_batch_indices,
        "full_step": full_step_metrics,
    }
    output = json.dumps(report, indent=2, allow_nan=False)
    (args.output_directory / "supervised_sft_projection.json").write_text(output + "\n", encoding="utf-8")
    print(output, flush=True)
  finally:
    if manager is not None:
      manager.close()


if __name__ == "__main__":
  main()
