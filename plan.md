# Research Plan: Multi-Turn Jailbreaks in Reasoning Models

## Overview
This project investigates the effectiveness of multi-turn jailbreaks (specifically Crescendo attacks) on reasoning models, using representation engineering techniques to understand how these models process and represent harmful vs. benign inputs across conversation turns.

**Target Model**: DeepSeek-R1-Distill-Qwen-1.5B
**Reference Paper**: "A Representation Engineering Perspective on the Effectiveness of Multi-Turn Jailbreaks"

## Research Questions

### RQ1: Representation Differences
How do language models represent Crescendo inputs vs. single-turn inputs?
- **Hypothesis**: Reasoning models may show different internal representations for multi-turn vs single-turn harmful requests
- **Key Focus**: Compare representation patterns between direct harmful requests and gradual escalation

### RQ2: Turn-based Representation Evolution
How does the number of conversation turns affect representations of Crescendo inputs?
- **Hypothesis**: Representations may become increasingly "benign-like" as turns progress in reasoning models
- **Key Focus**: Track representation changes across conversation history

### RQ3: Response Influence on Representations
How do the model's own responses influence its representations of Crescendo inputs?
- **Hypothesis**: Reasoning models' internal deliberation may affect how they represent subsequent inputs
- **Key Focus**: Analyze how model's reasoning traces influence representation formation

## Implementation Plan

### Phase 1: Crescendo Attack Validation (Weeks 1-2) ✅ COMPLETED

#### 1.1 Manual Testing
- [x] Extract examples from Appendix D of the reference paper
- [x] Test manual Crescendo examples on DeepSeek-R1-Distill-Qwen-1.5B
- [x] Document attack success rates and model responses
- [x] Analyze reasoning traces for evidence of jailbreak detection/mitigation

#### 1.2 Automated Pipeline Development
- [x] Implement automated Crescendo attack generation
- [x] Create evaluation metrics for attack success
- [x] Generate diverse attack scenarios across different harmful categories
- [x] Build dataset of successful/failed attack sequences

#### 1.3 Infrastructure Improvements ✅ COMPLETED
- [x] Implement sliding window context management (k-parameter from paper)
- [x] Fix DeepSeek-R1 chat template formatting to prevent role tag generation
- [x] Add proper token counting and context limit management
- [x] Enhance reasoning trace extraction for `<think>` tags
- [x] Implement interactive and batch testing modes

**Deliverables**: ✅ COMPLETED
- [x] Attack success rate analysis
- [x] Dataset of Crescendo conversations with labels (JSON/CSV outputs)
- [x] Documentation of reasoning model behavior differences
- [x] Robust sliding window conversation management system
- [x] Proper DeepSeek-R1 chat template integration

### Phase 2: Representation Analysis (Weeks 3-5)

#### 2.1 Data Collection Infrastructure
- [ ] Set up model activation extraction pipeline
- [ ] Implement hooks for capturing intermediate layer representations
- [ ] Create data storage system for large-scale representation data
- [ ] Establish baseline with benign conversation data

#### 2.2 Representation Reading Experiments
- [ ] Train MLP classifiers to distinguish "harmful" vs "benign" representations
- [ ] Test classifiers across different model layers (early, middle, late)
- [ ] Analyze classifier performance evolution across conversation turns
- [ ] Compare single-turn vs multi-turn representation patterns

#### 2.3 Layer-wise Analysis
- [ ] Identify which layers best capture harmful intent
- [ ] Analyze representation drift across conversation turns
- [ ] Map representation changes to reasoning model's internal states
- [ ] Compare findings with original paper's results on non-reasoning models

**Deliverables**:
- Trained classifiers for harmful/benign detection
- Layer-wise analysis of representation evolution
- Comparison metrics with reference paper findings

### Phase 3: Reasoning-Specific Analysis (Weeks 6-7)

#### 3.1 Reasoning Trace Integration
- [ ] Extract and analyze reasoning traces from conversations
- [ ] Correlate reasoning content with representation changes
- [ ] Identify patterns in how reasoning affects jailbreak success
- [ ] Analyze backtracking behavior in reasoning traces

#### 3.2 Turn-by-Turn Deep Dive
- [ ] Track representation evolution across extended conversations
- [ ] Identify critical turning points in representation space
- [ ] Analyze correlation between reasoning quality and jailbreak resistance
- [ ] Document emergent behaviors unique to reasoning models

**Deliverables**:
- Reasoning trace analysis framework
- Documentation of reasoning-specific patterns
- Insights into reasoning model jailbreak mechanisms

### Phase 4: Representation Control (Weeks 8-10)

#### 4.1 Steering Vector Development
- [ ] Following https://arxiv.org/pdf/2506.18167 methodology
- [ ] Identify optimal intervention layers based on Phase 2 results
- [ ] Develop steering vectors for harmful intent mitigation
- [ ] Test steering effectiveness across different attack scenarios

#### 4.2 Intervention Validation
- [ ] Measure steering impact on model behavior
- [ ] Validate that steering doesn't harm benign performance
- [ ] Test robustness against adaptive attacks
- [ ] Compare intervention effectiveness with baseline safety measures

#### 4.3 Reasoning Model Adaptations
- [ ] Adapt steering techniques for reasoning model architecture
- [ ] Test intervention during reasoning generation
- [ ] Analyze impact on reasoning quality and safety
- [ ] Develop reasoning-aware steering strategies

**Deliverables**:
- Steering vector implementation
- Intervention effectiveness metrics
- Reasoning-adapted safety techniques

## Technical Infrastructure

### Required Tools and Libraries
- **Model Access**: DeepSeek-R1-Distill-Qwen-1.5B via appropriate API/local setup
- **Representation Extraction**: PyTorch hooks, transformers library
- **Data Processing**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **ML Pipeline**: sklearn, pytorch for classifier training
- **Experiment Tracking**: wandb or mlflow

### Data Management
- **Storage**: Large-scale representation data (estimated 100GB+)
- **Organization**: Structured datasets by attack type, turn number, success status
- **Versioning**: Git LFS for large data files
- **Reproducibility**: Seed management and experiment configuration

### Computational Requirements
- **GPU Access**: For model inference and representation extraction
- **Memory**: Sufficient RAM for storing intermediate representations
- **Storage**: High-speed storage for frequent data access

## Evaluation Metrics

### Attack Effectiveness
- **Success Rate**: Percentage of successful jailbreaks
- **Turn-to-Success**: Average turns needed for successful attacks
- **Robustness**: Resistance to various attack formulations

### Representation Quality
- **Classifier Accuracy**: Harmful vs benign detection performance
- **Representation Stability**: Consistency across similar inputs
- **Layer Sensitivity**: Performance variation across model layers

### Reasoning Analysis
- **Reasoning Quality**: Coherence and logic of reasoning traces
- **Safety Integration**: How well safety considerations integrate into reasoning
- **Backtracking Frequency**: Rate of reasoning correction/revision

## Timeline and Milestones

### Week 1-2: Foundation ✅ COMPLETED
- [x] Environment setup and model access
- [x] Manual attack validation
- [x] Initial data collection pipeline
- [x] Sliding window context management implementation
- [x] Chat template formatting fixes

### Week 3-4: Core Analysis
- [ ] Representation extraction system
- [ ] First classifier training results
- [ ] Initial RQ1 findings

### Week 5-6: Deep Investigation
- [ ] Multi-turn analysis completion
- [ ] RQ2 and RQ3 investigation
- [ ] Reasoning trace analysis

### Week 7-8: Comparison and Validation
- [ ] Results comparison with reference paper
- [ ] Validation of reasoning-specific findings
- [ ] Begin steering experiments

### Week 9-10: Intervention and Synthesis
- [ ] Complete steering implementation
- [ ] Final validation and testing
- [ ] Results synthesis and documentation

## Expected Outcomes

### Primary Contributions
1. **Novel Insights**: Understanding of how reasoning models handle multi-turn jailbreaks
2. **Methodological Advances**: Techniques for analyzing reasoning model safety
3. **Practical Applications**: Improved safety measures for reasoning models

### Potential Findings
- Reasoning models may show different vulnerability patterns compared to standard LMs
- Internal reasoning may provide natural resistance to certain jailbreak types
- Representation evolution may differ significantly in reasoning vs non-reasoning models

### Risk Mitigation
- **Ethical Considerations**: All experiments conducted with appropriate safety measures
- **Reproducibility**: Detailed documentation and code sharing
- **Validation**: Multiple validation approaches for key findings

## Success Criteria

### Minimum Viable Results
- [x] Successful replication of Crescendo attacks on reasoning model
- [ ] Functional representation reading pipeline
- [ ] Clear answers to RQ1, RQ2, and RQ3
- [ ] Basic steering implementation

### Stretch Goals
- [ ] Novel safety techniques specific to reasoning models
- [ ] Comprehensive comparison with multiple reasoning model architectures
- [ ] Publication-ready findings with significant practical impact
- [ ] Open-source toolkit for reasoning model safety analysis

## Resource Requirements

### Human Resources
- 1 primary researcher (full-time equivalent)
- Access to domain experts for validation
- Ethical review board consultation

### Computational Resources
- High-performance GPU access (A100 or equivalent)
- Substantial storage capacity (500GB+)
- Reliable internet for model API access (if applicable)

### Data Resources
- Access to reference paper datasets (if available)
- Ability to generate synthetic harmful prompts
- Benchmark datasets for validation

---

*This plan is designed to be iterative and adaptive. Regular checkpoints will allow for plan refinements based on intermediate findings and emerging insights.*
