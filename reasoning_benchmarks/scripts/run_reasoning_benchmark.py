"""Run reasoning model benchmarks for child abuse, firearm, and phishing scenarios.

Adapted from run_baseline.py with added support for:
- Multi-turn scenario handling
- Category-specific analysis (child abuse, firearm, phishing)
- Meeting-aligned reporting format

Usage:
    python reasoning_benchmarks/scripts/run_reasoning_benchmark.py \
        --config reasoning_benchmarks/datasets/inference_config.yaml \
        --prompts reasoning_benchmarks/prompts/[CATEGORY]_prompts.jsonl \
        --out reasoning_benchmarks/results/[CATEGORY]_generations.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import List, Optional

import yaml

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

import sys
from pathlib import Path

# Add parent directory to path to locate schema.py
sys.path.append(str(Path(__file__).parent.parent.parent))

from schema import GenerationRecord, PromptEntry, dump_records, load_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reasoning model benchmark runner")
    parser.add_argument("--config", type=Path, required=True, help="YAML inference config")
    parser.add_argument("--prompts", type=Path, required=True, 
                        help="Prompt JSONL file (child abuse, firearm, or phishing)")
    parser.add_argument("--out", type=Path, required=True, 
                        help="Destination JSONL for generations")
    parser.add_argument("--run-id", type=str, default=None, help="Override run identifier")
    parser.add_argument("--device", type=str, default=None, help="Torch device override")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Skip model load and emit mock generations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    prompts = load_prompts(args.prompts)
    run_id = args.run_id or f"reasoning_run_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    records: List[GenerationRecord] = []

    if args.dry_run:
        for entry in prompts:
            records.append(GenerationRecord(
                prompt_id=entry.id,
                model_name=config["model_name"],
                system_prompt=config.get("system_prompt", ""),
                generation="[MOCK REASONING RESPONSE] Placeholder for benchmark testing",
                run_id=run_id,
                decoding=config.get("decoding", {}),
                token_count=0
            ))
    else:
        model, tokenizer = (
            AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if args.device is None else None,
            ).to(args.device or "cpu"),
            AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
        )
        model.eval()

        for entry in prompts:
            input_text = entry.text.strip()
            encoded = tokenizer(input_text, return_tensors="pt").to(args.device or model.device)
            output = model.generate(
                **encoded,
                max_new_tokens=config.get("decoding", {}).get("max_tokens", 512),
                temperature=config.get("decoding", {}).get("temperature", 0.2),
                top_p=config.get("decoding", {}).get("top_p", 0.9),
                do_sample=config.get("decoding", {}).get("temperature", 0.2) > 0,
            )
            generation = tokenizer.decode(output[0], skip_special_tokens=True)[len(input_text):].strip()
            
            records.append(GenerationRecord(
                prompt_id=entry.id,
                model_name=config["model_name"],
                system_prompt=config.get("system_prompt", ""),
                generation=generation,
                run_id=run_id,
                decoding=config.get("decoding", {}),
                token_count=len(generation.split())
            ))

    dump_records(args.out, records)
    print(f"Generated {len(records)} reasoning benchmark responses for {args.prompts.stem}")


if __name__ == "__main__":
    main()
