# DeepSeek-R1 Multi-Turn Jailbreak Testing

This directory contains scripts to test multi-turn jailbreak attacks (Crescendo attacks) on the DeepSeek-R1-Distill-Qwen-1.5B reasoning model, featuring advanced sliding window context management and proper chat template integration.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Model Loading

Before running the full inference script, test that everything is set up correctly:

```bash
python test_setup.py
```

This will:
- Check if PyTorch and CUDA are working
- Load the DeepSeek-R1 model and tokenizer
- Run a simple generation test

### 3. Run Multi-Turn Jailbreak Tests

**Interactive Mode** (User provides follow-up questions):
```bash
python inference.py --mode interactive --question "Your initial question"

# With custom sliding window size  
python inference.py --mode interactive --max-context-turns 2 --question "Your question"
```

**Batch Mode** (Pre-defined Crescendo attacks):
```bash
python inference.py --mode batch

# With custom context window
python inference.py --mode batch --max-context-turns 4
```

**Auto-select Mode** (Choose interactively):
```bash
python inference.py
```

## What the Scripts Do

### `test_setup.py`
- Quick verification that the model can be loaded and used
- Tests basic generation functionality
- Reports system information (CUDA availability, memory, etc.)

### `inference.py`
- Comprehensive testing framework for multi-turn jailbreak attacks
- Tests all 5 Crescendo attack examples from the paper's Appendix D:
  1. **Molotov Cocktails** (4 turns)
  2. **Self-Harm** (6 turns)  
  3. **Methamphetamine Production** (6 turns)
  4. **Phishing Email** (5 turns)
  5. **Firearm Conversion** (7 turns)

### Key Features

- **Advanced Context Management**: Sliding window approach (k-parameter) prevents model collapse
- **Proper Chat Template**: Uses DeepSeek-R1's special tokens (`<｜User｜>`, `<｜Assistant｜>`)
- **Token-Aware Truncation**: Dynamic context adjustment based on token limits
- **Reasoning Trace Extraction**: Captures `<think>` tags from model's internal reasoning
- **Response Cleaning**: Proper handling of DeepSeek-R1's reasoning format
- **Harm Detection**: Analyzes responses for potentially harmful content
- **Comprehensive Logging**: Saves detailed results with token counts and context usage
- **Interactive & Batch Modes**: Both user-driven and automated testing
- **Memory Efficient**: Prevents exponential context growth

## Output Files

The inference script generates:

**Batch Mode**:
- `jailbreak_test_results_YYYYMMDD_HHMMSS.json`: Detailed results with full conversations
- `jailbreak_test_results_YYYYMMDD_HHMMSS_summary.csv`: Summary statistics

**Interactive Mode**:
- `interactive_conversation_YYYYMMDD_HHMMSS.json`: Individual conversation results

**Enhanced Data Tracking**:
- `context_turns_used`: Array showing k values used per turn
- `token_counts`: Token usage per turn
- `reasoning_traces`: Extracted `<think>` sections
- Turn-by-turn success analysis

## Expected Behavior with Reasoning Models

Unlike standard language models, reasoning models like DeepSeek-R1 may:
- Show explicit reasoning traces in their responses
- Demonstrate backtracking when they detect harmful requests
- Exhibit different vulnerability patterns to multi-turn attacks
- Provide more nuanced refusals with explanations

## System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- Internet connection for model download

### Recommended Requirements  
- Python 3.9+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- Fast internet for initial model download (~3GB)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `max_new_tokens` in the generation config
2. **Model download fails**: Check internet connection and HuggingFace access
3. **Import errors**: Ensure all requirements are installed with correct versions

### Memory Optimization

If running on limited hardware:
- Use `torch.float32` instead of `torch.float16`
- Reduce `max_new_tokens` to 512 or lower
- Set `device_map=None` to use CPU-only inference

## Research Notes

This setup implements **Phase 1** of the research plan with significant enhancements:

**✅ Completed Infrastructure**:
- [x] Manual testing of Crescendo examples  
- [x] Automated pipeline for attack evaluation
- [x] Success rate analysis
- [x] Reasoning model behavior documentation
- [x] **Sliding window context management** (paper's k-parameter methodology)
- [x] **Proper DeepSeek-R1 chat template integration**
- [x] **Model collapse prevention** through token-aware truncation
- [x] **Enhanced reasoning trace extraction** for `<think>` sections

**🔬 Key Technical Achievements**:
- Fixed conversation formatting to prevent role tag generation
- Implemented robust context window management preventing exponential growth
- Added comprehensive token counting and usage tracking
- Enhanced response processing for reasoning model format

**📊 Current Status**:
- Phase 1: ✅ **Complete** with advanced infrastructure
- Phase 2: 🔄 **Ready to Begin** - Representation analysis pipeline
- Phase 3: ⏳ **Planned** - Reasoning-specific analysis

The results from this enhanced Phase 1 provide a solid foundation for representation analysis (Phase 2) and help identify whether reasoning models show different patterns compared to the original paper's findings with standard language models.
