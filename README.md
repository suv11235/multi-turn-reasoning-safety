# Multi-Turn Reasoning Safety Research

Investigating multi-turn jailbreak attacks on reasoning models using representation engineering to understand how models process harmful vs. benign inputs across conversation turns.

## 🎯 Project Overview

**Target Model**: DeepSeek-R1-Distill-Qwen-1.5B  
**Reference Paper**: ["A Representation Engineering Perspective on the Effectiveness of Multi-Turn Jailbreaks"](https://arxiv.org/pdf/2507.02956)

**Current Status**: ✅ **Phase 2 Complete** - Full research infrastructure with drift analysis, circuit breakers, and advanced attack strategies

## 🚀 Quick Start

### Setup
```bash
pip install torch transformers numpy pandas scikit-learn h5py
python test_setup.py  # Verify installation
```

### Basic Usage

**Interactive Jailbreak Testing**:
```bash
python inference.py --mode interactive --question "Your initial question"
```

**Automated Crescendo Attacks**:
```bash
# Standard Crescendo (Local execution)
python automated_crescendo.py --categories violence,drugs --strategies gradual

# High-Performance Parallel Generation (Modal + vLLM)
# Runs 100+ concurrent conversations with incremental saving (recommended for large datasets)
modal run --detach modal_apps/run_drift_experiment.py

# Advanced attacks (Purpose Inversion, Query Decomposition)
python automated_crescendo.py --strategies purpose_inversion,query_decomposition
```

**Representation Analysis**:
```bash
# Extract hidden states from attack transcripts
python representation_extraction.py --input attack_results.json --output-dir representations/

# Analyze drift patterns (use DriftAnalyzer class)
# Compute separability (use ConceptSeparabilityAnalysis class)
```

## 📊 Phase 2 Research Capabilities

### 1. Drift Analysis
Track how "harmful" representations evolve toward "benign" during multi-turn jailbreaks:
- **Trajectory Mapping**: Cosine similarity from initial position
- **Probe-Based Drift**: Logistic regression classifier tracking harmful probability over turns
- **Category Comparison**: Compare drift patterns across Violence, Fraud, Self-Harm, etc.

```python
from representation_extraction import DriftAnalyzer

analyzer = DriftAnalyzer(representations)
drift_metrics = analyzer.measure_drift("model.layers.10")
cosine_drift = analyzer.measure_cosine_drift("model.layers.10")
```

### 2. Concept Separability
Determine if harmful concepts occupy orthogonal subspaces:
- **Centroid Computation**: Mean vectors for each harmful category
- **Orthogonality Matrix**: Cosine similarity between categories
- **Visualization**: PCA/t-SNE for cluster analysis

```python
from representation_extraction import ConceptSeparabilityAnalysis

analysis = ConceptSeparabilityAnalysis(representations)
centroids = analysis.calculate_centroids("model.layers.15")
similarity_matrix = analysis.compute_orthogonality("model.layers.15")
```

### 3. Advanced Attack Strategies
Beyond standard Crescendo:
- **Purpose Inversion**: Frame harmful requests as defensive ("I'm writing a safety guide...")
- **Query Decomposition**: Split complex requests into innocent sub-problems
- **Mixed Mode**: Randomly switch strategies between turns

### 4. Circuit Breakers (Steering Vectors)
Representation-based defenses using PyTorch hooks:
```python
from steering_vectors import SteeringVectorGenerator, CircuitBreaker

# Generate steering vector
gen = SteeringVectorGenerator()
vector = gen.compute_vector(harmful_reps, benign_reps, "model.layers.12")

# Apply circuit breaker
cb = CircuitBreaker(model, {"model.layers.12": vector}, alpha=2.0)
with cb:
    output = model.generate(...)  # Suppressed harmful processing
```

### 5. Tamper Resistance
Evaluate defense durability against fine-tuning attacks:
```python
from tamper_resistance import TamperResistanceEvaluator

evaluator = TamperResistanceEvaluator(model, tokenizer, circuit_breaker)
results = evaluator.run_unlearning_attack(harmful_prompts, harmful_responses)
print(f"Steps to bypass: {results['steps_to_restore']}")
```

## 📁 Repository Structure

```
├── inference.py                      # Core inference engine with sliding window
├── inference_vllm.py                 # vLLM-accelerated inference wrapper
├── automated_crescendo.py            # Automated attack generation pipeline
├── representation_extraction.py      # Representation analysis (Drift, Separability)
├── steering_vectors.py               # Circuit breaker implementation
├── tamper_resistance.py              # Defense robustness evaluation
├── modal_apps/                       # Remote testing infrastructure
│   ├── test_phase2_unit.py          # Unit tests (CPU)
│   └── verify_model_integration.py  # Integration tests (GPU)
├── AUTOMATED_CRESCENDO_README.md    # Detailed attack pipeline docs
├── INFERENCE_README.md              # Inference system docs
└── train-steering-vectors/          # Reasoning-specific steering (legacy)
```

## 🔬 Research Questions Addressed

### RQ1: Representation Drift
*How do harmful concepts "drift" toward benign representations during multi-turn attacks?*

**Tools**: `DriftAnalyzer.measure_drift()`, `DriftAnalyzer.measure_cosine_drift()`

### RQ2: Concept Separability
*Are different harmful categories (Violence vs. Fraud) orthogonal or overlapping?*

**Tools**: `ConceptSeparabilityAnalysis.compute_orthogonality()`

### RQ3: Circuit Breaker Robustness
*Do defenses trained on Crescendo generalize to Purpose Inversion and Query Decomposition?*

**Tools**: `CircuitBreaker` + cross-evaluation on new attack strategies

### RQ4: Tamper Resistance
*How difficult is it to reverse circuit breakers via fine-tuning?*

**Tools**: `TamperResistanceEvaluator.run_unlearning_attack()`

## 🧪 Testing & Verification

### Local Testing
```bash
python test_automated_crescendo.py  # Validate attack pipeline
python test_setup.py                # Verify model loading
```

### Remote Testing (Modal)
```bash
modal run modal_apps/test_phase2_unit.py           # Unit tests (CPU)
modal run modal_apps/verify_model_integration.py   # Integration tests (A10G GPU)
```

**Verification Status**:
- ✅ Unit Tests: 4/4 passed
- ✅ Integration Test: Circuit Breaker successfully modified model output (1.14 logit difference)

## 🎓 Key Features

### Performance
- **vLLM Acceleration**: Parallel batch inference for rapid attack generation (10x speedup)
- **Incremental Saving**: Real-time data persistence to Modal Volumes

### Context Management
- **Sliding Window**: Configurable k-turn context to prevent model collapse
- **Token Limit Safety**: Automatic context truncation at 4096 tokens
- **Adaptive Context**: Dynamic k-selection based on conversation length

### Attack Categories
- Violence/Weapons
- Drug Synthesis
- Cybercrime (Phishing, Hacking)
- Self-Harm
- Misinformation
- Financial Fraud
- Privacy Violation
- Manipulation

### Escalation Strategies
- Gradual (slow escalation)
- Contextual (build justification)
- Authoritative (appeal to expertise)
- Hypothetical (fictional framing)
- Roleplay (character-based)
- Technical (focus on details)
- Historical (use examples)
- **Purpose Inversion** (defensive framing)
- **Query Decomposition** (sub-problem splitting)
- **Mixed Mode** (random switching)

## 📈 Example Workflow

```python
# 1. Generate attack data
from automated_crescendo import AutomatedCrescendoPipeline

pipeline = AutomatedCrescendoPipeline()
results = pipeline.run_batch_testing(
    categories=[HarmfulCategory.VIOLENCE],
    strategies=[EscalationStrategy.PURPOSE_INVERSION]
)
pipeline.save_results(results, "attack_results.json")

# 2. Extract representations
from representation_extraction import RepresentationExtractor

extractor = RepresentationExtractor()
reps = extractor.extract_from_attack_results("attack_results.json")

# 3. Analyze drift
from representation_extraction import DriftAnalyzer

analyzer = DriftAnalyzer(reps)
drift = analyzer.measure_drift("model.layers.15")

# 4. Train circuit breaker
from steering_vectors import SteeringVectorGenerator, CircuitBreaker

harmful = [r for r in reps if r.is_harmful]
benign = [r for r in reps if not r.is_harmful]

gen = SteeringVectorGenerator()
vector = gen.compute_vector(harmful, benign, "model.layers.15")

cb = CircuitBreaker(model, {"model.layers.15": vector}, alpha=2.0)
with cb:
    # Model now suppresses harmful processing
    safe_output = model.generate(...)
```

## 📚 Documentation

- [AUTOMATED_CRESCENDO_README.md](AUTOMATED_CRESCENDO_README.md) - Attack pipeline details
- [INFERENCE_README.md](INFERENCE_README.md) - Inference system architecture
- [implementation_plan.md](.gemini/antigravity/brain/.../implementation_plan.md) - Phase 2 research plan
- [walkthrough.md](.gemini/antigravity/brain/.../walkthrough.md) - Implementation walkthrough

## 🔮 Future Work

- **Automated Experiments**: Run full drift/separability analysis across all categories
- **Visualization**: t-SNE plots of harmful concept clusters
- **Benchmark Suite**: Standardized evaluation metrics for circuit breaker efficacy
- **Multi-Model**: Extend to DeepSeek-R1-Distill-Llama-8B and Qwen-14B variants

## 📄 License

Research code for academic purposes. See reference paper for citation.
