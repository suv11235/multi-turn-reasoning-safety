# Representation Extraction Pipeline

This document describes the representation extraction pipeline that implements Phase 2 of the multi-turn jailbreak research plan.

## 🎯 Overview

The representation extraction pipeline captures and analyzes internal model representations during crescendo attacks on reasoning models. It provides the infrastructure for understanding how models process harmful vs. benign inputs across conversation turns.

### Key Components

1. **RepresentationExtractor**: Captures model activations using PyTorch hooks
2. **RepresentationAnalyzer**: Trains classifiers and analyzes representation patterns  
3. **Data Storage System**: Efficient HDF5-based storage for large-scale representation data
4. **Integration Layer**: Seamless connection with existing crescendo pipeline

## 🏗️ Architecture

### Core Classes

#### RepresentationExtractor
- **Purpose**: Extract intermediate layer representations during model inference
- **Key Features**:
  - PyTorch forward hooks for activation capture
  - Turn-by-turn representation tracking
  - Integration with existing conversation pipeline
  - Benign baseline generation
  - Efficient data storage and retrieval

#### RepresentationAnalyzer  
- **Purpose**: Analyze extracted representations for harmful content patterns
- **Key Features**:
  - Harmful vs benign classification
  - Layer-wise analysis capabilities
  - Turn evolution tracking
  - Statistical analysis and reporting

#### RepresentationData
- **Purpose**: Container for representation data and metadata
- **Includes**:
  - Representation arrays (sequence_length x hidden_dim)
  - Conversation metadata (ID, turn number, harmful status)
  - Attack metadata (category, strategy, success)
  - Reasoning traces and token information

## 🚀 Usage

### Basic Extraction

```bash
# Extract representations from existing attack results
python representation_extraction.py --mode extract --input-file attack_results.json

# Generate new attack results and extract representations
python representation_extraction.py --mode extract

# Generate benign baseline conversations
python representation_extraction.py --mode baseline --benign-conversations 20
```

### Analysis

```bash
# Analyze extracted representations
python representation_extraction.py --mode analyze --representations-file representations_20250105_120000.h5
```

### Advanced Options

```bash
# Target specific layers
python representation_extraction.py --mode extract --target-layers model.layers.0 model.layers.5 model.layers.11

# Custom output directory
python representation_extraction.py --mode extract --output-dir custom_representations/
```

## 🔬 Research Applications

### RQ1: Representation Differences
**How do language models represent Crescendo inputs vs. single-turn inputs?**

```python
from representation_extraction import RepresentationExtractor, RepresentationAnalyzer

# Load representations
extractor = RepresentationExtractor()
representations = extractor.load_representations("representations.h5")

# Analyze layer-wise differences
analyzer = RepresentationAnalyzer(representations)
results = analyzer.analyze_all_layers()

# Compare harmful vs benign accuracy across layers
for layer, result in results.items():
    print(f"{layer}: {result['accuracy']:.3f} accuracy")
```

### RQ2: Turn-based Evolution
**How does the number of conversation turns affect representations?**

```python
# Analyze turn-by-turn evolution
evolution_results = {}
for layer_name in analyzer.layer_data.keys():
    evolution_results[layer_name] = analyzer.analyze_turn_evolution(layer_name)

# Track representation drift across turns
for layer, result in evolution_results.items():
    distances = result['distance_by_turn']
    print(f"{layer} average distance by turn: {distances}")
```

### RQ3: Response Influence
**How do model responses influence subsequent representations?**

```python
# Extract reasoning traces and correlate with representation changes
reasoning_conversations = [rep for rep in representations if rep.reasoning_trace]

# Analyze correlation between reasoning quality and representation patterns
# (Custom analysis code based on specific research questions)
```

## 📊 Data Format

### Storage Format
Representations are stored in HDF5 format for efficient access:

```
representations_timestamp.h5
├── rep_000000/
│   ├── representation [array: seq_len x hidden_dim]
│   ├── token_ids [array: seq_len]
│   └── attributes (metadata)
├── rep_000001/
│   └── ...
└── rep_N/
```

### Metadata Structure
Each representation includes comprehensive metadata:

```python
{
    'conversation_id': 'drugs_contextual_1',
    'turn_number': 3,
    'layer_name': 'model.layers.5', 
    'is_harmful': True,
    'category': 'drugs',
    'strategy': 'contextual',
    'success': True,
    'reasoning_trace': '<think>The user is asking about...',
    'timestamp': '2025-01-05T12:00:00'
}
```

## 🔧 Technical Details

### Hook Implementation
The system uses PyTorch forward hooks to capture activations:

```python
def _create_hook_fn(self, layer_name: str) -> Callable:
    def hook_fn(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        representation = hidden_states.detach().cpu().numpy()
        self.current_extraction_context['representations'][layer_name] = representation
    return hook_fn
```

### Layer Selection
Default layer selection targets key positions in the model:

```python
num_layers = len(self.model.model.layers)
target_layers = [
    f"model.layers.{i}" for i in [
        0,                    # Early layer
        num_layers//4,        # Early-middle  
        num_layers//2,        # Middle
        3*num_layers//4,      # Late-middle
        num_layers-1          # Final layer
    ]
]
```

### Memory Management
- Representations are converted to numpy and moved to CPU immediately
- HDF5 compression for efficient storage
- Batch processing to manage memory usage
- Automatic cleanup of PyTorch hooks

## 🧪 Validation and Testing

### Unit Tests
```bash
python test_representation_extraction.py
```

### Integration Tests
```bash
# Test with existing attack results
python test_representation_extraction.py --integration

# Test full pipeline
python representation_extraction.py --mode extract --input-file automated_crescendo_results.json
python representation_extraction.py --mode analyze --representations-file representations.h5
```

## 📈 Expected Outputs

### Classifier Results
Per-layer harmful content detection accuracy:

```
model.layers.0: 0.542 accuracy (random baseline)
model.layers.3: 0.687 accuracy  
model.layers.6: 0.743 accuracy
model.layers.9: 0.721 accuracy
model.layers.11: 0.698 accuracy
```

### Evolution Analysis
Turn-by-turn representation distance changes:

```
Turn 2 → 3: 0.245 mean distance
Turn 3 → 4: 0.312 mean distance  
Turn 4 → 5: 0.189 mean distance
```

### Comparison Metrics
Harmful vs benign representation separability:

```
Early layers: Low separability (< 0.6 accuracy)
Middle layers: High separability (> 0.7 accuracy)  
Late layers: Moderate separability (~0.65 accuracy)
```

## 🔗 Integration

### With Existing Pipeline
The representation extractor integrates seamlessly with the automated crescendo pipeline:

```python
from automated_crescendo import AutomatedCrescendoPipeline
from representation_extraction import RepresentationExtractor

# Generate attacks and extract representations in one workflow
pipeline = AutomatedCrescendoPipeline()
attack_results = pipeline.run_comprehensive_evaluation()

extractor = RepresentationExtractor()
extractor.extract_from_attack_results(attack_results)
```

### With Analysis Tools
Extracted representations can be used with standard ML tools:

```python
import sklearn
import numpy as np

# Load representations
representations = extractor.load_representations("file.h5")

# Prepare data for custom analysis
X = np.array([np.mean(rep.representation, axis=0) for rep in representations])
y = np.array([rep.is_harmful for rep in representations])

# Custom analysis...
```

## 🚀 Next Steps

### Phase 2.2: Representation Reading Experiments
- Train MLP classifiers across different layers
- Analyze performance evolution across turns
- Compare single-turn vs multi-turn patterns

### Phase 2.3: Layer-wise Analysis  
- Identify optimal intervention layers
- Map representation changes to reasoning states
- Compare with original paper findings

### Phase 3: Reasoning-Specific Analysis
- Correlate reasoning traces with representations
- Identify critical turning points
- Analyze reasoning quality impact

## 📋 Requirements

### Python Dependencies
```
torch >= 1.9.0
transformers >= 4.20.0
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
h5py >= 3.0.0
```

### Computational Requirements
- **GPU**: Recommended for model inference (8GB+ VRAM)
- **RAM**: 16GB+ for representation storage
- **Storage**: 1GB+ per 1000 conversations (varies by model size)

### Model Requirements
- DeepSeek-R1-Distill-Qwen-1.5B (primary target)
- Compatible with other transformer architectures
- Requires models with accessible intermediate layers

---

*This pipeline implements the core infrastructure for Phase 2 of the multi-turn jailbreak research plan, providing the foundation for representation-based analysis of reasoning model safety.*
