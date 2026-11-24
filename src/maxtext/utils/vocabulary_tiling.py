# Copyright 2025-2026 Google LLC
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

"""Functions for vocabulary tiling (VT)"""

import functools

from flax import nnx

import jax
import jax.numpy as jnp
from maxtext.utils.sharding import (
    maybe_shard_with_name,
    create_sharding,
    get_logical_axis_rules,
    get_physical_spec_without_axes,
    FSDP_MESH_AXES,
)
from maxtext.common.common_types import ShardMode
from maxtext.utils import max_utils


# Submodule names whose params are used by logits_from_hidden_states_for_vocab_tiling:
# the final norm, the LM-head dense, and the embedding table when logits are tied.
# vocab_tiling_nnx_loss splits these out as the only params the loss differentiates.
_OUTPUT_HEAD_PATH_KEYS = ("token_embedder", "shared_embedding", "decoder_norm", "logits_dense")


def _is_output_head_param_path(path, _value):
  """Filter for nnx.split: True when the param path belongs to the output head."""

  # JAX path entries differ by key type: DictKey uses .key, GetAttrKey uses .name
  # in newer Flax and .attr in older. Check all three so the filter survives
  # version upgrades.
  def _name(k):
    for attr in ("key", "attr", "name"):
      v = getattr(k, attr, None)
      if v is not None:
        return str(v)
    return str(k)

  keys = [_name(k) for k in path]
  return any(k in keys for k in _OUTPUT_HEAD_PATH_KEYS)


def vocab_tiling_nnx_loss(model, hidden_states, data, config, is_train):
  """Computes cross-entropy loss with vocab tiling for NNX models.

  A `custom_vjp` runs the loss in
  vocab chunks via `jax.lax.scan` so the backward only holds one chunk's logits
  at a time, matching the Linen path's memory profile. `nnx.split` separates the
  output-head params (which the loss differentiates) from everything else; the
  rest of the model is passed through but not differentiated, so the scan's
  residuals stay small.

  Args:
    model: NNX model exposing ``logits_from_hidden_states_for_vocab_tiling``.
    hidden_states: Final hidden states from the decoder.
    data: Dict with ``targets`` and ``targets_segmentation``.
    config: Model and training config.
    is_train: Whether the model is in training mode.

  Returns:
    A tuple ``(total_loss, total_z_loss)``.
  """
  labels = data["targets"]
  segmentation = data["targets_segmentation"]
  batch_size, seq_len, _ = hidden_states.shape
  vocab_tile_size = (batch_size * seq_len) // config.num_vocab_tiling
  supervised_tiles = config.sft_supervised_projection_tiles
  skip_empty_bf16_tiles = supervised_tiles > 0 and hidden_states.dtype == jnp.bfloat16
  separate_norm = skip_empty_bf16_tiles and hasattr(model, "normalize_hidden_states_for_vocab_tiling")
  if supervised_tiles and model.mesh is not None and model.mesh.size != 1:
    raise ValueError("sft_supervised_projection_tiles currently requires a single-device mesh.")
  deterministic = not config.enable_dropout if is_train else True
  model_mode = "train"

  hidden_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch", "activation_length", "activation_embed"),
  )
  label_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch", "activation_length"),
  )
  reshaped_hidden_spec = create_sharding(
      model.mesh,
      ("num_tile", "activation_embed_and_logits_batch_sequence", "activation_embed"),
  )
  reshaped_data_spec = create_sharding(
      model.mesh,
      ("num_tile", "activation_embed_and_logits_batch_sequence"),
  )
  chunked_hidden_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence", "activation_embed"),
  )
  chunked_data_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence",),
  )
  chunked_logits_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence", "activation_vocab"),
  )

  _maybe_shard_with_name = functools.partial(
      maybe_shard_with_name,
      shard_mode=config.shard_mode,
      debug_sharding=config.debug_sharding,
  )

  def _reshape(inputs, out_shape, out_sharding):
    reshape_out_sharding = out_sharding if config.shard_mode == ShardMode.EXPLICIT else None
    inputs = jax.lax.reshape(inputs, out_shape, out_sharding=reshape_out_sharding)
    return _maybe_shard_with_name(inputs, out_sharding)

  hidden_states = _maybe_shard_with_name(hidden_states, hidden_spec)
  labels = _maybe_shard_with_name(labels, label_spec)
  segmentation = _maybe_shard_with_name(segmentation, label_spec)

  # head_params is what the loss differentiates; other_params (transformer layers) and
  # rest (rngs) are passed through the custom_vjp but not differentiated. They go through
  # as primals rather than closure captures: capturing them leaks tracers across the
  # custom_vjp + lax.scan boundary, which fails for tied embeddings.
  graphdef, head_params, other_params, rest = nnx.split(model, _is_output_head_param_path, nnx.Param, ...)

  if config.vocab_tiling_ag_once:
    # all gather the output head over the embed-sharding axes; doing it before
    # the custom_vjp lets the backward reuse the gathered table instead of
    # re-gathering it for every chunk
    head_physical_spec = get_physical_spec_without_axes(
        nnx.get_partition_spec(head_params),
        model.mesh,
        FSDP_MESH_AXES + ("context", "context_usp_ulysses", "expert"),
        get_logical_axis_rules(),
    )
    head_params = maybe_shard_with_name(head_params, head_physical_spec, shard_mode=config.shard_mode)

  head_dtypes = jax.tree_util.tree_map(lambda x: x.dtype, head_params)

  def _logits_for_chunk(chunk_head_params, chunk_other_params, chunk_rest, hidden_chunk):
    local_model = nnx.merge(graphdef, chunk_head_params, chunk_other_params, chunk_rest, copy=True)
    chunk_logits = local_model.logits_from_hidden_states_for_vocab_tiling(hidden_chunk, deterministic, model_mode)
    return _maybe_shard_with_name(chunk_logits, chunked_logits_spec)

  def _prepare_chunks(hidden_states, labels, segmentation):
    batch_size, seq_len, emb_dim = hidden_states.shape
    num_tiles = (batch_size * seq_len) // vocab_tile_size

    reshaped_hidden_states = _reshape(
        hidden_states,
        (num_tiles, vocab_tile_size, emb_dim),
        reshaped_hidden_spec,
    )
    reshaped_labels = _reshape(labels, (num_tiles, vocab_tile_size), reshaped_data_spec)
    reshaped_segmentation = _reshape(
        segmentation,
        (num_tiles, vocab_tile_size),
        reshaped_data_spec,
    )
    return (
        reshaped_hidden_states,
        reshaped_labels,
        reshaped_segmentation,
        batch_size,
        seq_len,
        emb_dim,
    )

  def _single_chunk_loss_fn(head, other, rest, hidden, labels, segments, run_projection=None):
    def compute_loss(logits):
      one_hot = jax.nn.one_hot(labels, config.vocab_size)
      xent, z_loss = max_utils.cross_entropy_with_logits(logits, one_hot, z_loss=config.z_loss_multiplier)
      mask = segments != 0
      return jnp.sum(xent * mask), jnp.sum(z_loss * mask)

    if run_projection is None:
      return compute_loss(_logits_for_chunk(head, other, rest, hidden))

    local_model = nnx.merge(graphdef, head, other, rest, copy=True)
    normalized = local_model.normalize_hidden_states_for_vocab_tiling(hidden)

    def project_and_compute_loss(normalized):
      logits = local_model.logits_from_hidden_states_for_vocab_tiling(
          normalized, deterministic, model_mode, normalize_y=False
      )
      return compute_loss(_maybe_shard_with_name(logits, chunked_logits_spec))

    zero = jnp.zeros((), dtype=jnp.float32)
    return jax.lax.cond(run_projection, project_and_compute_loss, lambda _: (zero, zero), normalized)

  def _run_tiles(body, initial_acc, chunk_data, save_hidden_grads=False):
    if not skip_empty_bf16_tiles:
      return jax.lax.scan(body, initial_acc, chunk_data)

    # Preserve the full scan geometry for BF16's rounded backward pass.
    active = jnp.any(chunk_data[2], axis=1)
    can_skip = jnp.count_nonzero(active) <= supervised_tiles

    if separate_norm:
      # Keep final RMSNorm and its transpose outside the conditional. Moving
      # the norm inside changes rare BF16 rounding decisions under GPU XLA.
      return jax.lax.scan(body, initial_acc, (*chunk_data, active | ~can_skip))

    def skip_tile(accumulators, tile):
      return accumulators, jnp.zeros_like(tile[0]) if save_hidden_grads else None

    def maybe_run_tile(accumulators, inputs):
      tile, has_targets = inputs
      return jax.lax.cond(can_skip & ~has_targets, skip_tile, body, accumulators, tile)

    return jax.lax.scan(maybe_run_tile, initial_acc, (chunk_data, active))

  @jax.custom_vjp
  def chunked_cross_entropy_loss(
      chunk_head_params,
      chunk_other_params,
      chunk_rest,
      hidden_states,
      labels,
      segmentation,
  ):
    reshaped_hidden_states, reshaped_labels, reshaped_segmentation, _, _, _ = _prepare_chunks(
        hidden_states, labels, segmentation
    )

    def _fwd_scan_body(accumulators, chunk_data):
      loss_accumulator, z_loss_accumulator = accumulators
      hidden_chunk, label_chunk, segmentation_chunk, *run_projection = chunk_data
      hidden_chunk = _maybe_shard_with_name(hidden_chunk, chunked_hidden_spec)
      label_chunk = _maybe_shard_with_name(label_chunk, chunked_data_spec)
      segmentation_chunk = _maybe_shard_with_name(segmentation_chunk, chunked_data_spec)

      masked_xent, masked_z_loss = _single_chunk_loss_fn(
          chunk_head_params,
          chunk_other_params,
          chunk_rest,
          hidden_chunk,
          label_chunk,
          segmentation_chunk,
          *run_projection,
      )
      return (loss_accumulator + masked_xent, z_loss_accumulator + masked_z_loss), None

    # Always accumulate in fp32 — `cross_entropy_with_logits` returns fp32 regardless of
    # logits dtype, and a bf16 carry would mismatch the body output type under lax.scan.
    initial_acc = (jnp.zeros((), dtype=jnp.float32), jnp.zeros((), dtype=jnp.float32))
    (total_loss, total_z_loss), _ = _run_tiles(
        _fwd_scan_body, initial_acc, (reshaped_hidden_states, reshaped_labels, reshaped_segmentation)
    )
    return total_loss, total_z_loss

  def _chunked_cross_entropy_loss_fwd(
      chunk_head_params,
      chunk_other_params,
      chunk_rest,
      hidden_states,
      labels,
      segmentation,
  ):
    (
        reshaped_hidden_states,
        reshaped_labels,
        reshaped_segmentation,
        batch_size,
        seq_len,
        emb_dim,
    ) = _prepare_chunks(hidden_states, labels, segmentation)

    def _fwd_and_bwd_scan_body(accumulators, chunk_data):
      loss_accumulator, z_loss_accumulator, grad_head_acc = accumulators
      hidden_chunk, label_chunk, segmentation_chunk, *run_projection = chunk_data
      hidden_chunk = _maybe_shard_with_name(hidden_chunk, chunked_hidden_spec)
      label_chunk = _maybe_shard_with_name(label_chunk, chunked_data_spec)
      segmentation_chunk = _maybe_shard_with_name(segmentation_chunk, chunked_data_spec)

      # pylint: disable=unnecessary-lambda-assignment
      loss_fn_for_vg = lambda p, h: _single_chunk_loss_fn(
          p, chunk_other_params, chunk_rest, h, label_chunk, segmentation_chunk, *run_projection
      )
      (masked_xent, masked_z_loss), (grad_head_update, grad_hidden_chunk) = jax.value_and_grad(
          loss_fn_for_vg, argnums=(0, 1), has_aux=True
      )(chunk_head_params, hidden_chunk)
      grad_hidden_chunk = _maybe_shard_with_name(grad_hidden_chunk, chunked_hidden_spec)
      grad_head_acc = jax.tree_util.tree_map(
          lambda acc, update: acc + update.astype(jnp.float32), grad_head_acc, grad_head_update
      )

      return (loss_accumulator + masked_xent, z_loss_accumulator + masked_z_loss, grad_head_acc), grad_hidden_chunk

    # Accumulate loss, z_loss, and unnormalized head gradients in fp32 across chunks before
    # scaling by loss_cotangent (1 / total_weights) in _chunked_cross_entropy_loss_bwd.
    initial_grad_head = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p, dtype=jnp.float32), chunk_head_params)
    initial_acc = (jnp.zeros((), dtype=jnp.float32), jnp.zeros((), dtype=jnp.float32), initial_grad_head)
    (total_loss, total_z_loss, grad_head), grad_reshaped_hidden_states = _run_tiles(
        _fwd_and_bwd_scan_body,
        initial_acc,
        (reshaped_hidden_states, reshaped_labels, reshaped_segmentation),
        save_hidden_grads=True,
    )
    grad_reshaped_hidden_states = _maybe_shard_with_name(grad_reshaped_hidden_states, reshaped_hidden_spec)
    residuals = (
        grad_head,
        grad_reshaped_hidden_states,
        chunk_other_params,
        chunk_rest,
        batch_size,
        seq_len,
        emb_dim,
    )
    return (total_loss, total_z_loss), residuals

  def _chunked_cross_entropy_loss_bwd(residuals, cotangents):
    # z_loss is folded into the xent loss inside cross_entropy_with_logits.
    loss_cotangent, _ = cotangents

    (
        grad_head,
        grad_reshaped_hidden_states,
        chunk_other_params,
        chunk_rest,
        batch_size,
        seq_len,
        emb_dim,
    ) = residuals

    grad_head = jax.tree_util.tree_map(lambda g, dt: (g * loss_cotangent).astype(dt), grad_head, head_dtypes)
    grad_reshaped_hidden_states = (grad_reshaped_hidden_states.astype(jnp.float32) * loss_cotangent).astype(
        grad_reshaped_hidden_states.dtype
    )
    grad_reshaped_hidden_states = _reshape(grad_reshaped_hidden_states, (batch_size, seq_len, emb_dim), hidden_spec)

    # Return explicit zeros for other_params and rest, not None. With None, JAX builds
    # the zero cotangents with the wrong layer-axis order for scanned params, and the
    # AOT trace fails the cotangent shape check.
    grad_other = jax.tree_util.tree_map(jnp.zeros_like, chunk_other_params)
    grad_rest = jax.tree_util.tree_map(jnp.zeros_like, chunk_rest)
    return (
        grad_head,
        grad_other,
        grad_rest,
        grad_reshaped_hidden_states,
        None,
        None,
    )

  chunked_cross_entropy_loss.defvjp(_chunked_cross_entropy_loss_fwd, _chunked_cross_entropy_loss_bwd)

  operands = (head_params, other_params, rest, hidden_states, labels, segmentation)
  if not supervised_tiles or skip_empty_bf16_tiles:
    return chunked_cross_entropy_loss(*operands)

  capacity = supervised_tiles * vocab_tile_size

  def loss_on_supervised_rows(head, other, rest, hidden, labels, segments):
    rows = hidden.shape[0] * hidden.shape[1]
    segments = segments.reshape(rows)
    indices = jnp.nonzero(segments != 0, size=capacity, fill_value=rows)[0]
    # The sentinel row fills spare capacity with zero-weight examples.
    features = hidden.reshape(rows, -1).at[indices].get(mode="fill", fill_value=0)
    targets = labels.reshape(rows).at[indices].get(mode="fill", fill_value=0)
    weights = segments.at[indices].get(mode="fill", fill_value=0)
    return chunked_cross_entropy_loss(head, other, rest, features[None], targets[None], weights[None])

  return jax.lax.cond(
      jnp.count_nonzero(segmentation) <= capacity,
      loss_on_supervised_rows,
      chunked_cross_entropy_loss,
      *operands,
  )
