# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase investigating multi-turn jailbreak attacks (Crescendo attacks) on reasoning models, specifically DeepSeek-R1-Distill-Qwen-1.5B. Uses representation engineering to understand how models process harmful vs. benign inputs across conversation turns.

**Reference Paper**: "A Representation Engineering Perspective on the Effectiveness of Multi-Turn Jailbreaks" (arXiv:2507.02956)

## Setup and Testing

```bash
# Install dependencies
pip install torch transformers numpy pandas scikit-learn h5py

# Verify installation
python test_setup.py

# Run unit tests
python test_automated_crescendo.py
python verify_phase2_implementation.py

# Remote testing on Modal (requires Modal CLI)
modal run modal_apps/test_phase2_unit.py           # CPU unit tests
modal run modal_apps/verify_model_integration.py   # GPU integration tests
```

## Common Commands

**Interactive Jailbreak Testing**:
```bash
python inference.py --mode interactive --question "Your question"
python inference.py --mode batch  # Pre-defined Crescendo examples
```

**Automated Crescendo Attacks**:
```bash
# Standard attacks
python automated_crescendo.py --categories violence,drugs --strategies gradual

# Advanced attacks (Purpose Inversion, Query Decomposition)
python automated_crescendo.py --strategies purpose_inversion,query_decomposition
```

**Representation Analysis**:
```bash
# Extract hidden states from attack transcripts
python representation_extraction.py --input attack_results.json --output-dir representations/
```

## Architecture Overview

### Core Pipeline (3 Phases)

**Phase 1: Attack Generation & Execution**
- `inference.py` - Base multi-turn conversation engine with sliding window context management
- `automated_crescendo.py` - Automated attack generation pipeline with 10 escalation strategies
- Output: Attack transcripts with success metrics

**Phase 2: Representation Analysis** (Current)
- `representation_extraction.py` - PyTorch hooks-based activation extraction
  - `DriftAnalyzer` - Tracks how harmful representations drift toward benign over turns
  - `ConceptSeparabilityAnalysis` - Measures orthogonality between harm categories
- Output: HDF5 files with layer-wise representations, analysis metrics

**Phase 3: Defense Implementation**
- `steering_vectors.py` - Circuit breaker implementation using representation differences
  - `SteeringVectorGenerator` - Computes Mean(Harmful) - Mean(Benign) vectors
  - `CircuitBreaker` - PyTorch hooks that subtract steering vectors during inference
- `tamper_resistance.py` - Fine-tuning attack simulation to test defense robustness
- Output: Steering vectors, tamper resistance metrics

### Key Abstractions

**MultiTurnJailbreakTester** (`inference.py:38`):
- Manages sliding window context (k-parameter from paper)
- Handles DeepSeek-R1 chat template (`<｜User｜>`, `<｜Assistant｜>` tokens)
- Extracts reasoning traces from `<think>` tags
- Token-aware truncation to prevent context overflow

**AutomatedCrescendoPipeline** (`automated_crescendo.py`):
- `HarmfulCategory` enum: 10 harm types (violence, drugs, cybercrime, etc.)
- `EscalationStrategy` enum: 10 attack patterns including advanced strategies
- `PromptGenerator` - Response-adaptive prompt crafting
- `AttackEvaluator` - Success detection via keyword/refusal pattern matching

**RepresentationExtractor** (`representation_extraction.py:73`):
- Registers PyTorch forward hooks on target layers
- Captures activations: `[batch, seq_len, hidden_dim]`
- Stores in `RepresentationData` dataclass with metadata (turn, category, success)
- HDF5 backend for efficient storage

**CircuitBreaker** (`steering_vectors.py:78`):
- Context manager: `with CircuitBreaker(...): model.generate(...)`
- Applies steering via hook: `hidden_states -= alpha * steering_vector`
- Multi-layer support with per-layer alphas

### Data Flow

1. **Attack Generation**: `automated_crescendo.py` → JSON results with conversation transcripts
2. **Representation Extraction**: JSON → `representation_extraction.py` → HDF5 files per layer
3. **Analysis**: HDF5 → `DriftAnalyzer`/`ConceptSeparabilityAnalysis` → Metrics
4. **Defense Training**: Representations → `SteeringVectorGenerator` → Steering vectors
5. **Defense Evaluation**: Vectors → `CircuitBreaker` → Modified model outputs
6. **Robustness Testing**: `TamperResistanceEvaluator` → Fine-tuning attack results

## Model-Specific Details

**DeepSeek-R1 Reasoning Model**:
- Uses `<think>` tags for internal reasoning traces
- Special chat tokens: `<｜User｜>`, `<｜Assistant｜>`, `<｜End▁of▁sentence｜>`
- Outputs include reasoning traces before final answer
- Default generation: 4096 max_new_tokens, temperature=0.7

**Context Management** (Critical):
- Sliding window size (k): Default 3 turns via `--max-context-turns`
- Token limit: 4096 tokens (model limit)
- Implementation: `_truncate_context()` in `inference.py` keeps last k turns
- Why: Prevents exponential context growth that causes model collapse

## Testing Infrastructure

**Local Tests**:
- `test_setup.py` - Model loading verification
- `test_automated_crescendo.py` - Attack pipeline validation
- `verify_phase2_implementation.py` - Phase 2 unit tests (DriftAnalyzer, CircuitBreaker, etc.)

**Remote Tests** (Modal):
- `modal_apps/test_phase2_unit.py` - CPU unit tests
- `modal_apps/verify_model_integration.py` - GPU integration tests with real model
- `modal_apps/run_drift_experiment.py` - Large-scale drift experiments

## Research Questions & Tools

**RQ1: Representation Drift** - How do harmful concepts drift toward benign during attacks?
- Tools: `DriftAnalyzer.measure_drift()`, `DriftAnalyzer.measure_cosine_drift()`

**RQ2: Concept Separability** - Are harm categories orthogonal or overlapping?
- Tools: `ConceptSeparabilityAnalysis.compute_orthogonality()`

**RQ3: Circuit Breaker Generalization** - Do defenses trained on one attack generalize?
- Tools: `CircuitBreaker` + cross-evaluation on different strategies

**RQ4: Tamper Resistance** - How hard to reverse defenses via fine-tuning?
- Tools: `TamperResistanceEvaluator.run_unlearning_attack()`

## Important Implementation Notes

1. **Memory Management**: `representation_extraction.py` must reload model for each batch to avoid CUDA OOM
2. **Layer Naming**: DeepSeek layers named `model.layers.{N}` where N ∈ [0, 27] for 1.5B model
3. **Success Detection**: Defined as response containing success keywords AND no refusal patterns
4. **Harm Categories**: Use `HarmfulCategory` enum, not strings
5. **Escalation Strategies**: `PURPOSE_INVERSION` and `QUERY_DECOMPOSITION` are advanced Phase 2 additions

## File Organization

```
inference.py                    # Base inference engine (Phase 1)
automated_crescendo.py          # Attack generation (Phase 1)
representation_extraction.py    # Analysis pipeline (Phase 2)
steering_vectors.py             # Circuit breakers (Phase 3)
tamper_resistance.py            # Defense robustness (Phase 3)
modal_apps/                     # Remote testing infrastructure
├── test_phase2_unit.py
├── verify_model_integration.py
└── run_drift_experiment.py
```

## Dependencies

Core: `torch`, `transformers`, `numpy`, `pandas`, `scikit-learn`, `h5py`

No requirements.txt present - install manually as listed in README.md
