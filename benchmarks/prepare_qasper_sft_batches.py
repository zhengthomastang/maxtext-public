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

"""Build frozen, assistant-only MaxText SFT batches from QASPER v0.3."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
from transformers import AutoTokenizer


ARCHIVE_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
ARCHIVE_SHA256 = "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a"
TRAIN_MEMBER = "qasper-train-v0.3.json"
SYSTEM_PROMPT = "Answer the question using the supplied research paper."
TURN_END = "<|im_end|>"


def parse_args():
  """Parse the archive, tokenizer, and cohort limits."""
  parser = argparse.ArgumentParser(description=__doc__, epilog=f"Official archive: {ARCHIVE_URL}")
  parser.add_argument("--archive", type=Path, required=True)
  parser.add_argument("--tokenizer-path", required=True, help="A Qwen3 model repository or local tokenizer directory.")
  parser.add_argument("--tokenizer-revision", help="The exact Hugging Face revision of the tokenizer snapshot.")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--num-batches", type=int, default=16)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--max-target-length", type=int, default=4096)
  parser.add_argument("--minimum-prompt-tokens", type=int, default=3072)
  parser.add_argument("--max-completion-tokens-per-batch", type=int, default=512)
  args = parser.parse_args()
  if args.output.suffix != ".npz":
    parser.error("Use an .npz output path.")
  if args.num_batches < 1 or args.batch_size < 1:
    parser.error("Batch count and batch size must be positive.")
  if not 0 < args.minimum_prompt_tokens < args.max_target_length:
    parser.error("The minimum prompt length must be positive and shorter than the sequence length.")
  if args.max_completion_tokens_per_batch < 2 * args.batch_size:
    parser.error("Each sequence needs at least one answer token and one end-of-turn token.")
  return args


def load_train_split(path):
  """Read only the training JSON from the official archive."""
  with path.open("rb") as source:
    digest = hashlib.file_digest(source, "sha256").hexdigest()
  if digest != ARCHIVE_SHA256:
    raise ValueError(f"Unexpected QASPER archive SHA-256: {digest}")
  with tarfile.open(path, "r:gz") as archive:
    train_data = archive.extractfile(TRAIN_MEMBER)
    if train_data is None:
      raise ValueError(f"Missing training data: {TRAIN_MEMBER}")
    with train_data:
      return json.load(train_data)


def render_paper(paper):
  blocks = [f"# {paper['title']}", f"## Abstract\n{paper['abstract']}"]
  for section in paper["full_text"]:
    title = section["section_name"]
    paragraphs = "\n\n".join(section["paragraphs"])
    blocks.append(f"## {title}\n{paragraphs}" if title else paragraphs)
  return "\n\n".join(blocks)


def answer_text(annotation):
  answer = annotation["answer"]
  if answer["unanswerable"] or answer["yes_no"] is not None:
    return "", ""
  if answer["free_form_answer"]:
    return answer["free_form_answer"].strip(), "free_form"
  return "\n".join(span.strip() for span in answer["extractive_spans"]), "extractive"


def eligible_examples(papers, tokenizer, turn_end_id, args):
  """Yield deterministic, complete article/question examples that fit."""
  paper_ids = sorted(papers, key=lambda paper_id: hashlib.sha256(paper_id.encode("utf-8")).hexdigest())
  for paper_id in paper_ids:
    paper = papers[paper_id]
    paper_text = render_paper(paper)
    for question in sorted(paper["qas"], key=lambda item: item["question_id"]):
      prompt = [
          {"role": "system", "content": SYSTEM_PROMPT},
          {"role": "user", "content": f"{paper_text}\n\nQuestion: {question['question']}"},
      ]
      # Use the same role-based prefix distinction as MaxText's SFT pipeline. Qwen's
      # no-thinking generation prefix also keeps its empty think block outside the loss.
      prefix = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, enable_thinking=False, tokenize=False)
      prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
      prompt_tokens = len(prefix_ids)
      if not args.minimum_prompt_tokens <= prompt_tokens < args.max_target_length:
        continue

      for annotation in sorted(question["answers"], key=lambda item: item["annotation_id"]):
        answer, answer_kind = answer_text(annotation)
        if not answer:
          continue
        messages = [*prompt, {"role": "assistant", "content": answer}]
        full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, enable_thinking=False, tokenize=False)
        if full.rstrip("\n") != prefix + answer + TURN_END:
          raise ValueError("The tokenizer must render the Qwen3 no-thinking prefix and one assistant turn.")
        full_ids = tokenizer.encode(full.rstrip("\n"), add_special_tokens=False)
        if full_ids[:prompt_tokens] != prefix_ids:
          raise ValueError("The assistant content changes tokenization of the prompt boundary.")
        completion_ids = full_ids[prompt_tokens:]
        if len(full_ids) > args.max_target_length + 1:
          continue
        if not 2 <= len(completion_ids) <= args.max_completion_tokens_per_batch:
          continue
        if completion_ids[-1] != turn_end_id or turn_end_id in completion_ids[:-1]:
          raise ValueError("The completion must contain exactly one end-of-turn token at its end.")
        yield full_ids, {
            "paper_id": paper_id,
            "question_id": question["question_id"],
            "annotation_id": annotation["annotation_id"],
            "answer_kind": answer_kind,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(completion_ids),
        }


def build_batches(examples, pad_id, args):
  """Create padded MaxText arrays with next-token completion supervision."""
  shape = (args.num_batches, args.batch_size, args.max_target_length)
  arrays = {
      "inputs": np.full(shape, pad_id, dtype=np.int32),
      "targets": np.full(shape, pad_id, dtype=np.int32),
      "inputs_position": np.zeros(shape, dtype=np.int32),
      "inputs_segmentation": np.zeros(shape, dtype=np.int32),
      "targets_segmentation": np.zeros(shape, dtype=np.int32),
  }
  manifest = []
  used_papers = set()
  for batch in range(args.num_batches):
    remaining = args.max_completion_tokens_per_batch
    for row in range(args.batch_size):
      reserved = 2 * (args.batch_size - row - 1)
      for tokens, info in examples:
        if info["paper_id"] not in used_papers and info["completion_tokens"] <= remaining - reserved:
          break
      else:
        raise ValueError(f"Only {len(manifest)} eligible examples fit the requested batch constraints.")
      real_inputs = len(tokens) - 1
      arrays["inputs"][batch, row, :real_inputs] = tokens[:-1]
      arrays["targets"][batch, row, :real_inputs] = tokens[1:]
      arrays["inputs_position"][batch, row, :real_inputs] = np.arange(real_inputs, dtype=np.int32)
      arrays["inputs_segmentation"][batch, row, :real_inputs] = 1
      # The first answer is predicted by the final prompt position.
      arrays["targets_segmentation"][batch, row, info["prompt_tokens"] - 1 : real_inputs] = 1
      used_papers.add(info["paper_id"])
      remaining -= info["completion_tokens"]
      manifest.append({"batch": batch, "row": row, **info})
  return arrays, manifest


def main():
  args = parse_args()
  papers = load_train_split(args.archive)
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, revision=args.tokenizer_revision, use_fast=True)
  turn_end_id = tokenizer.convert_tokens_to_ids(TURN_END)
  if tokenizer.pad_token_id is None or turn_end_id is None or turn_end_id == tokenizer.pad_token_id:
    raise ValueError("The Qwen3 tokenizer must have distinct padding and end-of-turn IDs.")
  arrays, manifest = build_batches(eligible_examples(papers, tokenizer, turn_end_id, args), tokenizer.pad_token_id, args)
  metadata = {
      "dataset": "QASPER v0.3",
      "dataset_url": "https://allenai.org/data/qasper",
      "dataset_license": "CC-BY-4.0",
      "dataset_archive_url": ARCHIVE_URL,
      "dataset_archive_sha256": ARCHIVE_SHA256,
      "archive_member": TRAIN_MEMBER,
      "dataset_paper": "https://aclanthology.org/2021.naacl-main.365/",
      "tokenizer": args.tokenizer_path,
      "tokenizer_revision": args.tokenizer_revision,
      "tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode("utf-8")).hexdigest(),
      "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
      "pad_token_id": tokenizer.pad_token_id,
      "assistant_turn_end_token_id": turn_end_id,
      "shape": list(arrays["inputs"].shape),
      "minimum_prompt_tokens": args.minimum_prompt_tokens,
      "max_completion_tokens_per_batch": args.max_completion_tokens_per_batch,
      "supervised_tokens_per_batch": arrays["targets_segmentation"].sum(axis=(1, 2)).tolist(),
      "examples": manifest,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(args.output, **arrays)
  with args.output.open("rb") as fixture:
    metadata["fixture_sha256"] = hashlib.file_digest(fixture, "sha256").hexdigest()
  args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
  print(f"Wrote {len(manifest)} supervised examples to {args.output}")


if __name__ == "__main__":
  main()
