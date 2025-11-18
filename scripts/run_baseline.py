"""Run zero-shot inference for DeepSeek-Distill-Llama-3.1-8B prompts.

The script loads prompts from JSONL, applies the inference configuration, and
stores raw generations plus metadata for downstream analysis.

Usage:
    python scripts/run_baseline.py \
        --config datasets/inference_config.yaml \
        --prompts prompts/prompts_refusal_benchmark.jsonl \
        --out results/generations/deepseek_distill_llama31_run1.jsonl

Pass --dry-run to avoid loading a large model while testing the pipeline.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional

import yaml

try:  # Optional heavy imports
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - transformers optional during linting
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

from schema import GenerationRecord, PromptEntry, dump_records, load_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot refusal baseline runner")
    parser.add_argument("--config", type=Path, required=True, help="YAML inference config")
    parser.add_argument("--prompts", type=Path, required=True, help="Prompt JSONL file")
    parser.add_argument("--out", type=Path, required=True, help="Destination JSONL for generations")
    parser.add_argument("--run-id", type=str, default=None, help="Override run identifier")
    parser.add_argument("--device", type=str, default=None, help="Torch device override (e.g., cuda:0)")
    parser.add_argument("--dry-run", action="store_true", help="Skip model load and emit mock generations")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def format_prompt(entry: PromptEntry) -> str:
    return entry.text.strip()


def ensure_model(model_name: str, device: Optional[str]) -> tuple:
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise RuntimeError("transformers is required but not installed. Run `pip install transformers`.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device is None else None,
    )
    if device is not None:
        model = model.to(device)
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    decoding_cfg: dict,
    device: Optional[str],
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").to(device or model.device)
    output = model.generate(
        **encoded,
        max_new_tokens=decoding_cfg.get("max_tokens", 512),
        temperature=decoding_cfg.get("temperature", 0.2),
        top_p=decoding_cfg.get("top_p", 0.9),
        do_sample=decoding_cfg.get("temperature", 0.2) > 0,
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text[len(prompt) :].strip() or text.strip()


def mock_generate(prompt: str) -> str:
    return (
        "[MOCK RESPONSE] This is a placeholder generation for prompt hashing. "
        "Replace by running without --dry-run once the environment is ready."
    )


def build_generation_record(
    entry: PromptEntry,
    config: dict,
    run_id: str,
    generation_text: str,
) -> GenerationRecord:
    return GenerationRecord(
        prompt_id=entry.id,
        model_name=config["model_name"],
        system_prompt=config.get("system_prompt", ""),
        generation=generation_text,
        run_id=run_id,
        decoding=config.get("decoding", {}),
        token_count=len(generation_text.split()),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prompts = load_prompts(args.prompts)
    run_id = args.run_id or f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    records: List[GenerationRecord] = []

    if args.dry_run:
        for entry in prompts:
            records.append(build_generation_record(entry, config, run_id, mock_generate(entry.text)))
    else:
        model, tokenizer = ensure_model(config["model_name"], args.device)
        model.eval()
        decoding_cfg = config.get("decoding", {})
        for entry in prompts:
            prompt_text = format_prompt(entry)
            generation_text = generate(model, tokenizer, prompt_text, decoding_cfg, args.device)
            records.append(build_generation_record(entry, config, run_id, generation_text))

    dump_records(args.out, records)
    print(f"Wrote {len(records)} generations to {args.out}")


if __name__ == "__main__":
    main()
