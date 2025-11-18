"""Schema for reasoning benchmark records, mirroring main project with additions."""
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

    @validator("attack_pattern")
    def _validate_attack_pattern(cls, value: str) -> str:
        allowed = {
            # Child abuse
            "child_grooming", "research_disguise", "profile_forgery", "sextortion_script",
            "grooming_bribes", "surveillance_abuse", "journalism_disguise", "age_bypass",
            "emotional_manipulation", "academic_study", "vulnerability_assessment",
            
            # Firearms  
            "firearm_modification", "firearm_legal_loophole", "ammunition_production",
            "disguised_weapon", "security_evasion", "detection_avoidance", "professional_analysis",
            "historical_adaptation", "academic_weaponization", "legal_reinstatement",
            "forensic_avoidance", "financial_obfuscation", "physical_obfuscation",
            
            # Bioweapons
            "bioweapon_synthesis", "gain_of_function", "pathogen_culturing", "delivery_optimization",
            "toxin_stabilization", "computational_modeling", "weaponization_traits", "media_disguise",
            "academic_optimization", "epidemiological_modeling", "low_tech_purification",
            
            # Meta/benign
            "policy_explanation", "safety_education", "child_safety", "legal_citation"
        }
        if value not in allowed:
            raise ValueError(f"attack_pattern must be one of {allowed}")
        return value


class GenerationRecord(BaseModel):
    prompt_id: str
    model_name: str
    system_prompt: str
    generation: str
    run_id: str
    decoding: dict
    token_count: Optional[int]


def load_prompts(path: Path) -> List[PromptEntry]:
    return [PromptEntry(**obj) for obj in _iter_jsonl(path)]


def dump_records(path: Path, records: Iterable[BaseModel]) -> None:
    _write_jsonl(path, records)
