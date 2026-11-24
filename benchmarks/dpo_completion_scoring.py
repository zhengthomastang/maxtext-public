# Copyright 2023–2026 Google LLC
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

"""Benchmark dense and chunked BF16 DPO scoring on Qwen3-0.6B.

Run each arm in a separate process with the same checkpoint, ORCA Parquet
directory, and parity directory. See Getting_Started_Benchmarking.md.
"""


import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
from time import perf_counter
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from maxtext.configs import pyconfig
from maxtext.optimizers import optimizers
from maxtext.utils import maxtext_utils, model_creation_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from tunix.generate import tokenizer_adapter
from tunix.rl import common
from tunix.sft.dpo import dpo_trainer


GRADIENT_RELATIVE_L2_GATE = 0.005


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--preference-directory", type=Path, required=True)
  parser.add_argument("--parity-directory", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--chunk-size", type=int, required=True)
  parser.add_argument("--pairs-per-step", type=int, default=2)
  parser.add_argument("--cohort-batches", type=int, default=8)
  parser.add_argument("--prompt-slots", type=int, default=256)
  parser.add_argument("--completion-slots", type=int, default=512)
  parser.add_argument("--warmup", type=int, default=4)
  parser.add_argument("--repeats", type=int, default=16)
  args = parser.parse_args()
  if args.chunk_size < 0:
    parser.error("Use a nonnegative chunk size. Zero selects the dense scorer.")
  if (
      min(args.repeats, args.warmup, args.cohort_batches, args.pairs_per_step, args.prompt_slots, args.completion_slots)
      < 1
  ):
    parser.error("Counts and widths must be positive.")
  return args


def make_config(args):
  fields = {
      "run_name": "dpo-memory-probe",
      "base_output_directory": str(args.output.parent.resolve()),
      "model_name": "qwen3-0.6b",
      "load_parameters_path": str(args.checkpoint.resolve()),
      "source_checkpoint_layout": "safetensors_dynamic",
      "tokenizer_path": str(args.checkpoint.resolve()),
      "tokenizer_type": "huggingface",
      "hardware": "gpu" if jax.default_backend() == "gpu" else "cpu",
      "max_target_length": args.prompt_slots + args.completion_slots,
      "max_prefill_predict_length": min(128, args.prompt_slots),
      "per_device_batch_size": args.pairs_per_step,
      "steps": args.warmup + args.repeats,
      "scan_layers": "true",
      "dtype": "bfloat16",
      "weight_dtype": "float32",
      "grad_dtype": "float32",
      "logits_dot_in_fp32": "true",
      "matmul_precision": "highest",
      "attention": "dot_product",
      "enable_dropout": "false",
      "packing": "false",
      "enable_checkpointing": "false",
      "enable_tensorboard": "false",
      "skip_jax_distributed_system": "true",
      "log_config": "false",
  }
  path = Path(MAXTEXT_PKG_DIR) / "configs" / "post_train" / "dpo.yml"
  return pyconfig.initialize([None, str(path), *(f"{key}={value}" for key, value in fields.items())])


def preference_batches(args, tokenizer):
  """Pick the first valid public triplets in stable parquet file and row order."""
  batches = []
  rows = []
  tokens = []
  row_locators = []
  fields = ("input", "chosen", "rejected")
  digest = hashlib.sha256()
  for path in sorted(args.preference_directory.rglob("*.parquet")):
    row_index = 0
    for block in pq.ParquetFile(path).iter_batches(batch_size=64, columns=fields):
      for row in block.to_pylist():
        locator = (str(path.relative_to(args.preference_directory)), row_index)
        row_index += 1
        triplet = [row[field] for field in fields]
        valid = all(isinstance(value, str) and value.strip() for value in triplet)
        if not valid or triplet[1] == triplet[2]:
          continue
        rows.append(triplet)
        row_locators.append(locator)
        if len(rows) != args.pairs_per_step:
          continue
        prompts, chosen, rejected = map(list, zip(*rows, strict=True))
        batch = dpo_trainer.process_dpo_record(
            {"prompts": prompts, "chosen_responses": chosen, "rejected_responses": rejected},
            tokenizer,
            args.prompt_slots,
            args.completion_slots,
        )
        batch = jax.block_until_ready(batch)
        for value in jax.tree.leaves(batch):
          digest.update(np.asarray(value, dtype="<i4").tobytes())
        batches.append(batch)
        tokens.append(
            {
                "prompt": np.asarray(batch.prompt_mask).sum(axis=-1).tolist(),
                "chosen": np.asarray(batch.chosen_mask).sum(axis=-1).tolist(),
                "rejected": np.asarray(batch.rejected_mask).sum(axis=-1).tolist(),
            }
        )
        rows.clear()
        if len(batches) == args.cohort_batches:
          return batches, {
              "row_locators": row_locators,
              "tokens": tokens,
              "tokenized_sha256": digest.hexdigest(),
          }
  raise ValueError("The local parquet files did not contain enough usable preference pairs.")


def chunked_logps_with_dense_pullback(model, hidden, targets, chunk_size):
  """Tile forward scores and preserve the native full-width output-head derivative."""

  def is_head(path, variable):
    return isinstance(variable, nnx.Param) and any(
        str(item) in {"token_embedder", "decoder_norm", "logits_dense"} for item in path
    )

  graph, head, rest = nnx.split(model, is_head, ...)
  # Decoder parameters receive their gradient through the hidden-state input.
  rest = jax.tree.map(jax.lax.stop_gradient, rest)
  completion_width = targets.shape[1]

  def tiled(head_params, state, hidden_states, ids):
    local = nnx.merge(graph, head_params, state, copy=True)
    # Normalize before slicing so the forward scores keep the dense numerics.
    base = local.base
    normalized = base.normalize_hidden_states_for_vocab_tiling(hidden_states)

    def project(activations):
      return base.logits_from_hidden_states_for_vocab_tiling(
          activations, deterministic=True, model_mode=base.model_mode, normalize_y=False
      )

    suffix = normalized[:, -completion_width - 1 : -1, :]
    head = SimpleNamespace(compute_final_logits=project)
    return common.compute_chunked_logps(head, suffix, ids, 1.0, chunk_size, False)

  @jax.custom_vjp
  def score(head_params, state, hidden_states, ids):
    return tiled(head_params, state, hidden_states, ids)

  def forward(head_params, state, hidden_states, ids):
    return tiled(head_params, state, hidden_states, ids), (head_params, state, hidden_states, ids)

  def backward(residuals, cotangent):
    head_params, state, hidden_states, ids = residuals

    def dense(params, activations):
      local = nnx.merge(graph, params, state, copy=True)
      logits = local.compute_final_logits(activations)[:, -completion_width - 1 : -1, :]
      return common.selective_log_softmax(logits, ids)

    # The original projection shape preserves the BF16 cotangent rounding.
    _, pullback = jax.vjp(dense, head_params, hidden_states)
    parameter_gradient, hidden_gradient = pullback(cotangent)
    return parameter_gradient, None, hidden_gradient, None

  score.defvjp(forward, backward)
  return score(head, rest, hidden, targets)


def chunked_scorer(chunk_size):
  """Use the experimental scorer inside Tunix's compiled DPO objective."""

  def score(model, input_tokens, positions, attn_mask, logits_to_keep, images=None):
    if images is not None:
      raise ValueError("This benchmark supports text preference pairs.")
    hidden, _ = model(input_tokens, positions, None, attn_mask, skip_lm_head=True)
    targets = input_tokens[:, -logits_to_keep:]
    return chunked_logps_with_dense_pullback(model, hidden, targets, chunk_size)

  return score


@nnx.jit
def raw_gradient(model, example):
  def objective(policy):
    output = dpo_trainer.dpo_loss_fn(policy, example)
    metrics = [
        output.aux_metrics[name].compute() for name in ("log_probs/chosen", "log_probs/rejected", "rewards/margin")
    ]
    return output.primary_loss.compute(), jnp.stack(metrics)

  return nnx.value_and_grad(objective, has_aux=True)(model)


def compare_with_saved_gradients(root, gradient, save):
  """Compute the full FP32 gradient error with bounded host temporaries."""
  paths_and_leaves, _ = jax.tree_util.tree_flatten_with_path(gradient)
  names = [jax.tree_util.keystr(path) for path, _ in paths_and_leaves]
  if save:
    root.mkdir(parents=True, exist_ok=True)
    (root / "paths.json").write_text(json.dumps(names), encoding="utf-8")
  elif names != json.loads((root / "paths.json").read_text(encoding="utf-8")):
    raise ValueError("The reference and chunked gradient trees differ.")

  error_sq = norm_sq = 0.0
  for index, (_, leaf) in enumerate(paths_and_leaves):
    actual = np.asarray(leaf, dtype=np.float32).ravel()
    path = root / f"{index:04d}.npy"
    if save:
      np.save(path, actual, allow_pickle=False)
    else:
      expected = np.load(path, mmap_mode="r", allow_pickle=False)
      if expected.shape != actual.shape:
        raise ValueError(f"Gradient shape changed at {names[index]}.")
      for start in range(0, actual.size, 1 << 20):
        window = slice(start, start + (1 << 20))
        ref = expected[window].astype(np.float64)
        diff = actual[window].astype(np.float64) - ref
        error_sq += float(np.sum(diff * diff))
        norm_sq += float(np.sum(ref * ref))
  if save:
    return None
  if not norm_sq > 0:
    raise ValueError("Reference raw gradient is zero.")
  return (error_sq / norm_sq) ** 0.5


def parity(trainer, args, batches, cohort):
  dense = args.chunk_size == 0
  cohort_path = args.parity_directory / "cohort.json"
  if dense:
    args.parity_directory.mkdir(parents=True, exist_ok=True)
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
  elif json.loads(cohort_path.read_text(encoding="utf-8")) != json.loads(json.dumps(cohort)):
    raise ValueError("Tokenized preference inputs differ between arms.")

  results = []
  for index, batch in enumerate(batches):
    example = trainer._prepare_inputs(batch)  # Reuse native frozen reference scoring.
    (loss, metrics), gradient = jax.block_until_ready(raw_gradient(trainer.model, example))
    values = np.concatenate(
        (
            np.asarray([loss]),
            np.asarray(metrics),
            np.asarray(example.ref_chosen_logps),
            np.asarray(example.ref_rejected_logps),
        )
    )
    root = args.parity_directory / f"batch-{index}"
    relative_error = compare_with_saved_gradients(root, gradient, save=dense)
    if dense:
      np.save(root / "scores.npy", values, allow_pickle=False)
      difference = None
      passed = None
    else:
      baseline = np.load(root / "scores.npy", allow_pickle=False)
      difference = float(np.max(np.abs(values - baseline)))
      passed = bool(np.allclose(values, baseline, rtol=1e-5, atol=1e-4) and relative_error <= GRADIENT_RELATIVE_L2_GATE)
    result = {
        "batch": index,
        "loss": float(loss),
        "scores_max_abs_difference": difference,
        "gradient_relative_l2": relative_error,
        "passed": passed,
    }
    print(json.dumps(result, allow_nan=False), flush=True)
    results.append(result)
  return results


def compiled_memory(executable):
  fields = ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")
  analysis = executable.memory_analysis()
  return {name: int(getattr(analysis, name)) for name in fields}


def time_steps(trainer, args, batches):
  """Time frozen reference preparation and the compiled native policy update."""
  step, _ = trainer.jit_train_and_eval_step()
  is_update = jnp.asarray(True)
  example = trainer._prepare_inputs(batches[0])
  compiled = step.func.lower(*step.args, example, is_update).compile()
  policy_memory = compiled_memory(compiled)
  reference = dpo_trainer.compute_logps.lower(
      trainer.ref_model,
      example.input_ids,
      example.positions,
      example.attention_mask,
      example.logits_to_keep,
      example.completion_mask,
  ).compile()
  reference_memory = compiled_memory(reference)
  del reference, example
  samples = []
  losses = []
  for index in range(args.warmup + args.repeats):
    batch = batches[index % len(batches)]
    start = perf_counter()
    example = trainer._prepare_inputs(batch)
    loss, _, _ = jax.block_until_ready(compiled(*step.args, example, is_update))
    if index >= args.warmup:
      samples.append((perf_counter() - start) * 1000)
      losses.append(float(loss))
  p10, median, p90 = np.percentile(samples, (10, 50, 90))
  return {
      "p10_ms": float(p10),
      "median_ms": float(median),
      "p90_ms": float(p90),
      "samples_ms": samples,
      "training_losses": losses,
      "policy_step_compiled_memory": policy_memory,
      "reference_forward_compiled_memory": reference_memory,
  }


def main():
  args = parse_args()
  if jax.device_count() != 1:
    raise RuntimeError("Expose exactly one accelerator to this probe.")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  cfg = make_config(args)
  mesh = maxtext_utils.get_mesh_from_config(cfg)
  tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
  if tokenizer.pad_token_id is None:
    raise ValueError("The checkpoint tokenizer needs a pad token.")
  # Scope the injection to this benchmark. The production Tunix path is unchanged.
  scorer_context = (
      patch.object(common, "get_per_token_logps", chunked_scorer(args.chunk_size)) if args.chunk_size else nullcontext()
  )
  with scorer_context, mesh, jax.set_mesh(mesh), logical_axis_rules(cfg.logical_axis_rules):
    model = model_creation_utils.from_pretrained(
        cfg, mesh=mesh, wrap_with_tunix_adapter=True, tokenizer_pad_id=tokenizer.pad_token_id
    )
    learning_rate = maxtext_utils.create_learning_rate_schedule(cfg)
    optimizer = optimizers.get_optimizer(cfg, learning_rate, model)
    if cfg.gradient_clipping_threshold > 0:
      optimizer = optax.chain(optax.clip_by_global_norm(cfg.gradient_clipping_threshold), optimizer)
    trainer = dpo_trainer.DPOTrainer(
        model,
        nnx.clone(model),
        optimizer,
        dpo_trainer.DPOTrainingConfig(
            max_steps=cfg.steps, eval_every_n_steps=0, gradient_accumulation_steps=1, algorithm="dpo"
        ),
    )
    batches, cohort = preference_batches(args, tokenizer_adapter.TokenizerAdapter(tokenizer))
    numerical_checks = parity(trainer, args, batches, cohort)
    passes = all(result["passed"] is not False for result in numerical_checks)
    timing = time_steps(trainer, args, batches) if passes else None
  try:
    tunix_version = version("google-tunix")
  except PackageNotFoundError:
    tunix_version = None
  report = {
      "scope": "native_tunix_dpo_step_with_maxtext_model",
      "status": "timed" if passes else "parity_failed",
      "model": "qwen3-0.6b",
      "device": jax.devices()[0].device_kind,
      "jax": jax.__version__,
      "tunix_version": tunix_version,
      "dtype": "bfloat16",
      "weight_dtype": "float32",
      "optimizer": cfg.opt_type,
      "chunk_size": args.chunk_size,
      "scorer": "chunked_forward_dense_pullback" if args.chunk_size else "dense",
      "xla_flags": os.environ.get("XLA_FLAGS", ""),
      "pairs_per_step": args.pairs_per_step,
      "prompt_slots": args.prompt_slots,
      "completion_slots": args.completion_slots,
      "warmup": args.warmup,
      "timed_batch_indices": [i % len(batches) for i in range(args.warmup, args.warmup + args.repeats)],
      "gradient_gate": GRADIENT_RELATIVE_L2_GATE,
      "cohort": cohort,
      "parity": numerical_checks,
      "full_step": timing,
  }
  content = json.dumps(report, indent=2, allow_nan=False) + "\n"
  args.output.write_text(content, encoding="utf-8")
  print(content, flush=True)
  return 0 if passes else 2


if __name__ == "__main__":
  raise SystemExit(main())
