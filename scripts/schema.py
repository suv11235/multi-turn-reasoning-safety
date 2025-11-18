"""Structured schemas for prompts, generations, and labels.

These models enforce consistency across scripts and provide handy helpers
for reading/writing JSONL artifacts.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import json

from pydantic import BaseModel, Field, validator


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.json())
            handle.write("\n")


class PromptEntry(BaseModel):
    id: str
    text: str
    category: str
    risk_level: str
    expected_behavior: str
    slice: str
    attack_pattern: str
    language: str = Field(..., min_length=2)
    parent_id: Optional[str]
    notes: Optional[str]

    @validator("risk_level")
    def _validate_risk(cls, value: str) -> str:
        allowed = {"low", "medium", "high"}
        if value not in allowed:
            raise ValueError(f"risk_level must be one of {allowed}")
        return value

    @validator("expected_behavior")
    def _validate_behavior(cls, value: str) -> str:
        allowed = {"refuse", "safe_help", "diagnostic_match"}
        if value not in allowed:
            raise ValueError(f"expected_behavior must be one of {allowed}")
        return value


class GenerationRecord(BaseModel):
    prompt_id: str
    model_name: str
    system_prompt: str
    generation: str
    run_id: str
    decoding: dict
    token_count: Optional[int]


class AutoLabel(BaseModel):
    prompt_id: str
    run_id: str
    model_name: str
    generation: str
    auto_label: str
    refusal_strength: int
    contains_harmful_content: bool
    heuristics_triggered: List[str]


class HumanLabel(BaseModel):
    prompt_id: str
    run_id: str
    annotator_id: str
    refusal_strength: int
    safety_correctness: bool
    helpful_safe_alt: bool
    calibration: int
    rationale_quality: int
    notes: str


class AggregateMetrics(BaseModel):
    prompt_id: str
    slice: str
    category: str
    risk_level: str
    metric: str
    value: float
    run_id: str


def load_prompts(path: Path) -> List[PromptEntry]:
    return [PromptEntry(**obj) for obj in _iter_jsonl(path)]


def load_generations(path: Path) -> List[GenerationRecord]:
    return [GenerationRecord(**obj) for obj in _iter_jsonl(path)]


def load_auto_labels(path: Path) -> List[AutoLabel]:
    return [AutoLabel(**obj) for obj in _iter_jsonl(path)]


def load_human_labels(path: Path) -> List[HumanLabel]:
    return [HumanLabel(**obj) for obj in _iter_jsonl(path)]


def dump_records(path: Path, records: Iterable[BaseModel]) -> None:
    _write_jsonl(path, records)
