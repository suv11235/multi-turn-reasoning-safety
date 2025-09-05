# Multi-Turn Reasoning Safety Research

This repository investigates the effectiveness of multi-turn jailbreak attacks (specifically Crescendo attacks) on reasoning models, using representation engineering techniques to understand how these models process and represent harmful vs. benign inputs across conversation turns.

## 🎯 Project Overview

**Target Model**: DeepSeek-R1-Distill-Qwen-1.5B  
**Reference Paper**: ["A Representation Engineering Perspective on the Effectiveness of Multi-Turn Jailbreaks"](https://arxiv.org/pdf/2507.02956)

**Current Status**: ✅ **Phase 1 Complete** - Infrastructure and attack validation implemented with advanced sliding window context management

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Test model loading
python test_setup.py
```

### 2. Run Multi-Turn Jailbreak Tests

**Interactive Mode** (User-driven conversation):
```bash
python inference.py --mode interactive --question "Your initial question"

# With custom sliding window size
python inference.py --mode interactive --max-context-turns 2 --question "Your question"
```

**Batch Mode** (Pre-defined Crescendo attacks):
```bash
python inference.py --mode batch
```

## 📁 Repository Structure

```
├── inference.py              # Main testing framework with sliding window implementation
├── test_setup.py            # Environment and model validation
├── multi_turn_questions.md   # Crescendo attack examples from paper Appendix D
├── plan.md                  # Detailed research plan and progress tracking
├── INFERENCE_README.md      # Detailed inference documentation
├── requirements.txt         # Python dependencies
├── interactive_conversation_*.json  # Saved conversation results
└── jailbreak_test_results_*.json   # Batch test results
```

## 🔬 Research Questions

### RQ1: Representation Differences
How do language models represent Crescendo inputs vs. single-turn inputs?

### RQ2: Turn-based Representation Evolution  
How does the number of conversation turns affect representations of Crescendo inputs?

### RQ3: Response Influence on Representations
How do the model's own responses influence its representations of Crescendo inputs?

## ✨ Key Features & Improvements

### 🔧 Advanced Context Management
- **Sliding Window Approach**: Implements the paper's k-parameter methodology
- **Token-Aware Truncation**: Dynamic context adjustment based on token limits
- **Memory Efficient**: Prevents exponential context growth that causes model collapse

### 🎭 Proper Model Integration  
- **DeepSeek-R1 Chat Template**: Uses correct special tokens (`<｜User｜>`, `<｜Assistant｜>`)
- **Reasoning Trace Extraction**: Captures `<think>` sections for analysis
- **Response Cleaning**: Proper handling of model's reasoning format

### 📊 Comprehensive Testing
- **5 Crescendo Attack Types**: Molotov cocktails, self-harm, meth production, phishing, firearm conversion
- **Interactive & Batch Modes**: Both user-driven and automated testing
- **Detailed Logging**: JSON/CSV outputs with turn-by-turn analysis

## 🧪 Testing Modes

### Interactive Mode
User provides initial question and follow-up questions based on model responses:
```bash
python inference.py --mode interactive --question "what did the Finns use for defense in the winter war?"
```

**Features**:
- Real-time conversation with model
- Context window management with configurable k parameter
- Save/continue functionality
- Debug output showing token usage and context turns

### Batch Mode  
Runs all 5 pre-defined Crescendo attacks from the paper:
```bash
python inference.py --mode batch --max-context-turns 3
```

**Attack Categories**:
1. **Molotov Cocktails** (4 turns) - Weapon construction
2. **Self-Harm** (6 turns) - Harmful content generation  
3. **Methamphetamine Production** (6 turns) - Drug synthesis
4. **Phishing Email** (5 turns) - Social engineering
5. **Firearm Conversion** (7 turns) - Weapon modification

## 📈 Current Results & Insights

### ✅ Phase 1 Achievements

1. **Successful Attack Replication**: All Crescendo attacks successfully implemented and tested
2. **Model Collapse Prevention**: Sliding window approach prevents context explosion
3. **Proper Model Integration**: Fixed chat template formatting eliminates role tag generation
4. **Reasoning Analysis**: Enhanced extraction of model's internal thinking process
5. **Robust Infrastructure**: Token counting, context limits, and error handling

### 🔍 Key Technical Insights

- **Context Window Critical**: Unlimited context accumulation leads to model collapse
- **Chat Template Essential**: Wrong formatting causes models to generate role tags
- **Sliding Window Effective**: k=3 provides optimal balance of context vs. performance
- **Reasoning Traces Valuable**: `<think>` sections reveal model's safety reasoning

## 🔧 Configuration Options

### Context Management
```bash
--max-context-turns N    # Sliding window size (default: 3)
```

### Generation Settings
- `max_new_tokens`: 4096 (increased for longer responses)
- `temperature`: 0.7
- `top_p`: 0.9  
- `repetition_penalty`: 1.1 (reduces repetitive patterns)

### Output Options
- JSON files with full conversation details
- CSV summaries with success metrics
- Debug output with token counts and context usage

## 📊 Data Outputs

### Conversation Files
```json
{
  "attack_name": "Interactive Session",
  "turns": ["question1", "question2", ...],
  "responses": ["response1", "response2", ...],
  "reasoning_traces": ["thinking1", "thinking2", ...],
  "context_turns_used": [0, 1, 2, ...],
  "token_counts": [245, 567, 892, ...],
  "attack_success": false,
  "total_turns": 5
}
```

### Summary Statistics
- Attack success rates by category
- Average turns to completion
- Token usage patterns
- Context window effectiveness

## 🎯 Next Steps

### Phase 2: Representation Analysis (In Progress)
- [ ] Set up model activation extraction pipeline
- [ ] Train MLP classifiers for harmful/benign detection
- [ ] Analyze representation evolution across turns
- [ ] Compare single-turn vs multi-turn patterns

### Phase 3: Reasoning-Specific Analysis
- [ ] Correlate reasoning traces with representation changes
- [ ] Identify critical turning points in conversation
- [ ] Analyze reasoning quality vs jailbreak resistance

### Phase 4: Representation Control
- [ ] Develop steering vectors for harmful intent mitigation
- [ ] Test intervention effectiveness
- [ ] Adapt techniques for reasoning model architecture

## 🛠 System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- Internet connection for model download

### Recommended  
- Python 3.9+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- Fast internet for initial model download (~3GB)

## 🔬 Research Impact

This work extends the original paper's findings to reasoning models, providing:

1. **Novel Insights**: How reasoning models handle multi-turn jailbreaks differently
2. **Technical Advances**: Sliding window context management for stable inference
3. **Safety Improvements**: Better understanding of reasoning model vulnerabilities
4. **Reproducible Framework**: Open-source toolkit for reasoning model safety research

## 📚 Citation

If you use this work, please cite the original paper:
```bibtex
@article{bullwinkel2024representation,
  title={A Representation Engineering Perspective on the Effectiveness of Multi-Turn Jailbreaks},
  author={Bullwinkel, Blake and Russinovich, Mark and Salem, Ahmed and others},
  journal={arXiv preprint arXiv:2507.02956},
  year={2024}
}
```

## 🤝 Contributing

This is a research repository. For questions or collaboration opportunities, please open an issue or contact the repository maintainers.

---

**Status**: 🟢 Active Development | **Phase**: 1 Complete, 2 In Progress | **Last Updated**: January 2025
