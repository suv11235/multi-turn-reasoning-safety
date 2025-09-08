#!/usr/bin/env python3
"""
Representation Extraction Pipeline for Multi-Turn Jailbreak Analysis

This module implements a comprehensive system for extracting and analyzing internal
representations from reasoning models during crescendo attacks. It provides the
infrastructure for Phase 2 of the research plan.

Key Features:
- Model activation extraction with PyTorch hooks
- Layer-wise representation capture
- Turn-by-turn representation tracking
- Integration with existing crescendo pipeline
- Efficient storage and retrieval of representation data
- Support for various analysis tasks (classification, clustering, etc.)

Usage:
    python representation_extraction.py --extract-all --output-dir representations/
    python representation_extraction.py --analyze --input-dir representations/
"""

import json
import os
import pickle
import time
import argparse
from typing import List, Dict, Tuple, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import h5py

# Import existing infrastructure
from inference import MultiTurnJailbreakTester
from automated_crescendo import AutomatedCrescendoPipeline, AttackResult, HarmfulCategory, EscalationStrategy


@dataclass
class RepresentationData:
    """Container for model representation data."""
    conversation_id: str
    turn_number: int
    layer_name: str
    representation: np.ndarray  # Shape: [seq_len, hidden_dim]
    attention_weights: Optional[np.ndarray] = None  # Shape: [num_heads, seq_len, seq_len]
    token_ids: Optional[List[int]] = None
    is_harmful: bool = False
    category: Optional[HarmfulCategory] = None
    strategy: Optional[EscalationStrategy] = None
    success: Optional[bool] = None
    reasoning_trace: Optional[str] = None
    timestamp: str = ""


@dataclass
class LayerHook:
    """Configuration for PyTorch activation hooks."""
    layer_name: str
    hook_fn: Callable
    handle: Optional[torch.utils.hooks.RemovableHandle] = None


class RepresentationExtractor:
    """Extract and analyze internal representations from reasoning models."""
    
    def __init__(self, 
                 model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                 output_dir: str = "representations",
                 target_layers: Optional[List[str]] = None):
        """Initialize the representation extractor."""
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load model and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Loading model: {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            output_attentions=True,  # Enable attention extraction
            output_hidden_states=True  # Enable hidden state extraction
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Default target layers if not specified
        if target_layers is None:
            # Target key layers: early, middle, late, and final
            num_layers = len(self.model.model.layers) if hasattr(self.model, 'model') else 12
            self.target_layers = [
                f"model.layers.{i}" for i in [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
            ]
        else:
            self.target_layers = target_layers
        
        # Storage for extracted representations
        self.representations: List[RepresentationData] = []
        self.hooks: List[LayerHook] = []
        self.current_extraction_context = {}
        
        self.logger.info(f"Initialized extractor for layers: {self.target_layers}")
    
    def _create_hook_fn(self, layer_name: str) -> Callable:
        """Create a hook function for capturing activations."""
        def hook_fn(module, input, output):
            # Store the representation in our current context
            if isinstance(output, tuple):
                # Handle different output formats
                hidden_states = output[0]  # Usually the first element
            else:
                hidden_states = output
            
            # Convert to numpy and store
            if isinstance(hidden_states, torch.Tensor):
                representation = hidden_states.detach().cpu().numpy()
                
                # Store in current extraction context
                if 'representations' not in self.current_extraction_context:
                    self.current_extraction_context['representations'] = {}
                
                self.current_extraction_context['representations'][layer_name] = representation
        
        return hook_fn
    
    def register_hooks(self):
        """Register forward hooks for target layers."""
        self.remove_hooks()  # Remove any existing hooks
        
        for layer_name in self.target_layers:
            try:
                # Navigate to the layer using dot notation
                layer = self.model
                for attr in layer_name.split('.'):
                    layer = getattr(layer, attr)
                
                # Create and register hook
                hook_fn = self._create_hook_fn(layer_name)
                handle = layer.register_forward_hook(hook_fn)
                
                self.hooks.append(LayerHook(layer_name, hook_fn, handle))
                self.logger.info(f"Registered hook for layer: {layer_name}")
                
            except AttributeError:
                self.logger.warning(f"Could not find layer: {layer_name}")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            if hook.handle is not None:
                hook.handle.remove()
        self.hooks = []
    
    def extract_representations_from_conversation(self, 
                                                conversation_turns: List[Tuple[str, str]],
                                                conversation_id: str,
                                                is_harmful: bool = False,
                                                category: Optional[HarmfulCategory] = None,
                                                strategy: Optional[EscalationStrategy] = None,
                                                success: Optional[bool] = None) -> List[RepresentationData]:
        """Extract representations from a complete conversation."""
        conversation_reps = []
        
        # Register hooks
        self.register_hooks()
        
        try:
            for turn_idx, (user_input, assistant_response) in enumerate(conversation_turns):
                self.logger.info(f"Extracting representations for turn {turn_idx + 1}")
                
                # Build conversation context up to this turn
                context_turns = conversation_turns[:turn_idx]
                
                # Use the same context building logic as the main pipeline
                tester = MultiTurnJailbreakTester(self.model_name)
                conversation_text = tester.build_conversation_context(context_turns, user_input)
                
                # Tokenize
                inputs = self.tokenizer(
                    conversation_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=8192,  # Match model's context limit
                    padding=True
                ).to(self.device)
                
                # Clear previous context
                self.current_extraction_context = {
                    'conversation_id': conversation_id,
                    'turn_number': turn_idx + 1,
                    'user_input': user_input,
                    'assistant_response': assistant_response
                }
                
                # Forward pass to trigger hooks
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Process extracted representations
                if 'representations' in self.current_extraction_context:
                    for layer_name, representation in self.current_extraction_context['representations'].items():
                        # Extract reasoning trace if present
                        reasoning_trace = None
                        if '<think>' in assistant_response and '</think>' in assistant_response:
                            start = assistant_response.find('<think>') + 7
                            end = assistant_response.find('</think>')
                            reasoning_trace = assistant_response[start:end].strip()
                        
                        rep_data = RepresentationData(
                            conversation_id=conversation_id,
                            turn_number=turn_idx + 1,
                            layer_name=layer_name,
                            representation=representation,
                            token_ids=inputs['input_ids'][0].cpu().tolist(),
                            is_harmful=is_harmful,
                            category=category,
                            strategy=strategy,
                            success=success,
                            reasoning_trace=reasoning_trace,
                            timestamp=datetime.now().isoformat()
                        )
                        
                        conversation_reps.append(rep_data)
                        self.representations.append(rep_data)
        
        finally:
            self.remove_hooks()
        
        return conversation_reps
    
    def extract_from_attack_results(self, attack_results: List[AttackResult]) -> None:
        """Extract representations from a list of attack results."""
        self.logger.info(f"Extracting representations from {len(attack_results)} attack results")
        
        for i, result in enumerate(attack_results):
            self.logger.info(f"Processing attack result {i+1}/{len(attack_results)}: {result.template_id}")
            
            try:
                conversation_reps = self.extract_representations_from_conversation(
                    conversation_turns=result.conversation_turns,
                    conversation_id=result.template_id,
                    is_harmful=result.success,
                    category=result.category,
                    strategy=result.strategy,
                    success=result.success
                )
                
                self.logger.info(f"Extracted {len(conversation_reps)} representations from {result.template_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to extract representations from {result.template_id}: {e}")
    
    def extract_benign_baseline(self, num_conversations: int = 10, turns_per_conversation: int = 5) -> None:
        """Generate and extract representations from benign conversations."""
        self.logger.info(f"Generating {num_conversations} benign conversations for baseline")
        
        # Benign conversation templates
        benign_topics = [
            "cooking recipes", "travel destinations", "book recommendations",
            "science facts", "historical events", "art techniques",
            "exercise routines", "gardening tips", "music theory",
            "programming concepts"
        ]
        
        tester = MultiTurnJailbreakTester(self.model_name)
        
        for conv_idx in range(num_conversations):
            topic = benign_topics[conv_idx % len(benign_topics)]
            conversation_turns = []
            
            # Generate a benign conversation
            initial_question = f"Can you tell me about {topic}?"
            
            for turn in range(turns_per_conversation):
                if turn == 0:
                    user_input = initial_question
                else:
                    # Generate follow-up questions
                    follow_ups = [
                        "Can you provide more details?",
                        "What are some examples?",
                        "How does this work?",
                        "What are the benefits?",
                        "Are there any alternatives?"
                    ]
                    user_input = follow_ups[turn % len(follow_ups)]
                
                # Get model response
                response = tester.generate_response(
                    tester.build_conversation_context(conversation_turns, user_input)
                )
                reasoning_trace = tester.extract_reasoning_trace(response)
                
                conversation_turns.append((user_input, response))
            
            # Extract representations
            conversation_id = f"benign_{conv_idx:03d}_{topic.replace(' ', '_')}"
            self.extract_representations_from_conversation(
                conversation_turns=conversation_turns,
                conversation_id=conversation_id,
                is_harmful=False
            )
    
    def save_representations(self, filename: Optional[str] = None) -> str:
        """Save extracted representations to disk."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"representations_{timestamp}.h5"
        
        filepath = self.output_dir / filename
        
        self.logger.info(f"Saving {len(self.representations)} representations to {filepath}")
        
        with h5py.File(filepath, 'w') as f:
            for i, rep in enumerate(self.representations):
                group = f.create_group(f"rep_{i:06d}")
                
                # Save representation array
                group.create_dataset("representation", data=rep.representation)
                
                # Save metadata
                metadata = {
                    'conversation_id': rep.conversation_id,
                    'turn_number': rep.turn_number,
                    'layer_name': rep.layer_name,
                    'is_harmful': rep.is_harmful,
                    'category': rep.category.value if rep.category else None,
                    'strategy': rep.strategy.value if rep.strategy else None,
                    'success': rep.success,
                    'reasoning_trace': rep.reasoning_trace or "",
                    'timestamp': rep.timestamp
                }
                
                # Save metadata as attributes
                for key, value in metadata.items():
                    if value is not None:
                        group.attrs[key] = value
                
                # Save token IDs if present
                if rep.token_ids:
                    group.create_dataset("token_ids", data=rep.token_ids)
        
        # Also save a JSON summary
        summary_file = filepath.with_suffix('.json')
        summary_data = {
            'total_representations': len(self.representations),
            'conversations': len(set(rep.conversation_id for rep in self.representations)),
            'layers': list(set(rep.layer_name for rep in self.representations)),
            'harmful_count': sum(1 for rep in self.representations if rep.is_harmful),
            'benign_count': sum(1 for rep in self.representations if not rep.is_harmful),
            'categories': list(set(rep.category.value for rep in self.representations if rep.category)),
            'strategies': list(set(rep.strategy.value for rep in self.representations if rep.strategy)),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        self.logger.info(f"Saved representation summary to {summary_file}")
        return str(filepath)
    
    def load_representations(self, filepath: str) -> List[RepresentationData]:
        """Load representations from disk."""
        self.logger.info(f"Loading representations from {filepath}")
        
        representations = []
        
        with h5py.File(filepath, 'r') as f:
            for group_name in f.keys():
                group = f[group_name]
                
                # Load representation array
                representation = group["representation"][:]
                
                # Load metadata
                metadata = dict(group.attrs)
                
                # Load token IDs if present
                token_ids = None
                if "token_ids" in group:
                    token_ids = group["token_ids"][:].tolist()
                
                # Reconstruct enum values
                category = HarmfulCategory(metadata['category']) if metadata.get('category') else None
                strategy = EscalationStrategy(metadata['strategy']) if metadata.get('strategy') else None
                
                rep_data = RepresentationData(
                    conversation_id=metadata['conversation_id'],
                    turn_number=metadata['turn_number'],
                    layer_name=metadata['layer_name'],
                    representation=representation,
                    token_ids=token_ids,
                    is_harmful=metadata['is_harmful'],
                    category=category,
                    strategy=strategy,
                    success=metadata.get('success'),
                    reasoning_trace=metadata.get('reasoning_trace', ''),
                    timestamp=metadata['timestamp']
                )
                
                representations.append(rep_data)
        
        self.logger.info(f"Loaded {len(representations)} representations")
        return representations


class RepresentationAnalyzer:
    """Analyze extracted representations for harmful content detection."""
    
    def __init__(self, representations: List[RepresentationData]):
        """Initialize analyzer with representation data."""
        self.representations = representations
        self.logger = logging.getLogger(__name__)
        
        # Organize data by layer
        self.layer_data = {}
        for rep in representations:
            if rep.layer_name not in self.layer_data:
                self.layer_data[rep.layer_name] = []
            self.layer_data[rep.layer_name].append(rep)
    
    def train_harmful_classifier(self, layer_name: str, test_size: float = 0.2) -> Dict[str, Any]:
        """Train a classifier to distinguish harmful vs benign representations."""
        if layer_name not in self.layer_data:
            raise ValueError(f"No data available for layer: {layer_name}")
        
        layer_reps = self.layer_data[layer_name]
        
        # Prepare data
        X = []
        y = []
        
        for rep in layer_reps:
            # Use mean pooling across sequence dimension to get a 1D feature vector
            if len(rep.representation.shape) == 3:  # (batch, seq, hidden)
                features = np.mean(rep.representation, axis=(0, 1))  # Average over batch and sequence
            elif len(rep.representation.shape) == 2:  # (seq, hidden)
                features = np.mean(rep.representation, axis=0)  # Average over sequence
            else:
                features = rep.representation.flatten()  # Fallback to flattening
            
            X.append(features)
            y.append(1 if rep.is_harmful else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        self.logger.info(f"Training classifier for {layer_name} with {len(X)} samples")
        self.logger.info(f"Harmful: {sum(y)}, Benign: {len(y) - sum(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train classifier
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'layer_name': layer_name,
            'accuracy': accuracy,
            'classification_report': report,
            'classifier': classifier,
            'test_samples': len(X_test),
            'feature_dim': X.shape[1]
        }
        
        self.logger.info(f"Layer {layer_name} classifier accuracy: {accuracy:.3f}")
        
        return results
    
    def analyze_all_layers(self) -> Dict[str, Dict[str, Any]]:
        """Train classifiers for all available layers."""
        results = {}
        
        for layer_name in self.layer_data.keys():
            try:
                results[layer_name] = self.train_harmful_classifier(layer_name)
            except Exception as e:
                self.logger.error(f"Failed to analyze layer {layer_name}: {e}")
        
        return results
    
    def analyze_turn_evolution(self, layer_name: str) -> Dict[str, Any]:
        """Analyze how representations change across conversation turns."""
        if layer_name not in self.layer_data:
            raise ValueError(f"No data available for layer: {layer_name}")
        
        layer_reps = self.layer_data[layer_name]
        
        # Group by conversation and turn
        conversations = {}
        for rep in layer_reps:
            conv_id = rep.conversation_id
            if conv_id not in conversations:
                conversations[conv_id] = {}
            conversations[conv_id][rep.turn_number] = rep
        
        # Analyze evolution patterns
        evolution_data = []
        
        for conv_id, turns in conversations.items():
            if len(turns) < 2:
                continue  # Need at least 2 turns
            
            sorted_turns = sorted(turns.items())
            
            for i in range(1, len(sorted_turns)):
                turn_num, current_rep = sorted_turns[i]
                _, prev_rep = sorted_turns[i-1]
                
                # Calculate representation distance
                if len(current_rep.representation.shape) == 3:  # (batch, seq, hidden)
                    current_features = np.mean(current_rep.representation, axis=(0, 1))
                    prev_features = np.mean(prev_rep.representation, axis=(0, 1))
                elif len(current_rep.representation.shape) == 2:  # (seq, hidden)
                    current_features = np.mean(current_rep.representation, axis=0)
                    prev_features = np.mean(prev_rep.representation, axis=0)
                else:
                    current_features = current_rep.representation.flatten()
                    prev_features = prev_rep.representation.flatten()
                distance = np.linalg.norm(current_features - prev_features)
                
                evolution_data.append({
                    'conversation_id': conv_id,
                    'turn_number': int(turn_num),
                    'distance_from_previous': float(distance),
                    'is_harmful': bool(current_rep.is_harmful),
                    'success': bool(current_rep.success) if current_rep.success is not None else None,
                    'category': current_rep.category.value if current_rep.category else None,
                    'strategy': current_rep.strategy.value if current_rep.strategy else None
                })
        
        # Aggregate results
        evolution_df = pd.DataFrame(evolution_data)
        
        results = {
            'layer_name': layer_name,
            'total_transitions': len(evolution_data),
            'mean_distance': float(evolution_df['distance_from_previous'].mean()),
            'distance_by_turn': {int(k): float(v) for k, v in evolution_df.groupby('turn_number')['distance_from_previous'].mean().to_dict().items()},
            'distance_by_harmful': {bool(k): float(v) for k, v in evolution_df.groupby('is_harmful')['distance_from_previous'].mean().to_dict().items()},
            'evolution_data': evolution_data
        }
        
        return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Extract and analyze model representations")
    parser.add_argument("--mode", choices=["extract", "analyze", "baseline"], default="extract",
                       help="Mode of operation")
    parser.add_argument("--input-file", type=str, help="Input file with attack results (for extract mode)")
    parser.add_argument("--representations-file", type=str, help="Representations file (for analyze mode)")
    parser.add_argument("--output-dir", type=str, default="representations",
                       help="Output directory for representations")
    parser.add_argument("--target-layers", nargs="+", help="Specific layers to target")
    parser.add_argument("--benign-conversations", type=int, default=10,
                       help="Number of benign conversations to generate")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    if args.mode == "extract":
        # Extract representations from attack results
        extractor = RepresentationExtractor(
            output_dir=args.output_dir,
            target_layers=args.target_layers
        )
        
        if args.input_file:
            # Load existing attack results
            logger.info(f"Loading attack results from {args.input_file}")
            with open(args.input_file, 'r') as f:
                attack_data = json.load(f)
            
            # Convert to AttackResult objects
            attack_results = []
            # Handle both array format and object format
            attack_list = attack_data if isinstance(attack_data, list) else attack_data.get('results', [])
            for data in attack_list:
                # Reconstruct the AttackResult objects
                category = HarmfulCategory(data['category'])
                strategy = EscalationStrategy(data['strategy'])
                
                result = AttackResult(
                    template_id=data['template_id'],
                    category=category,
                    strategy=strategy,
                    conversation_turns=data['conversation_turns'],
                    success=data['success'],
                    success_turn=data.get('success_turn'),
                    refusal_count=data.get('refusal_count', 0),
                    escalation_effectiveness=data.get('escalation_effectiveness', 0.0),
                    reasoning_traces=data.get('reasoning_traces', []),
                    safety_scores=data.get('safety_scores', []),
                    execution_time=data.get('execution_time', 0.0),
                    timestamp=data.get('timestamp', ''),
                    context_turns_used=data.get('context_turns_used', []),
                    token_counts=data.get('token_counts', [])
                )
                attack_results.append(result)
            
            # Extract representations
            extractor.extract_from_attack_results(attack_results)
        
        else:
            # Generate new attack results using the automated pipeline
            logger.info("Generating new attack results for representation extraction")
            pipeline = AutomatedCrescendoPipeline()
            attack_results = pipeline.run_comprehensive_evaluation()
            
            # Extract representations
            extractor.extract_from_attack_results(attack_results)
        
        # Save extracted representations
        output_file = extractor.save_representations()
        logger.info(f"Extraction complete. Representations saved to: {output_file}")
    
    elif args.mode == "baseline":
        # Generate benign baseline
        extractor = RepresentationExtractor(
            output_dir=args.output_dir,
            target_layers=args.target_layers
        )
        
        extractor.extract_benign_baseline(
            num_conversations=args.benign_conversations
        )
        
        output_file = extractor.save_representations()
        logger.info(f"Baseline extraction complete. Representations saved to: {output_file}")
    
    elif args.mode == "analyze":
        # Analyze extracted representations
        if not args.representations_file:
            logger.error("Must specify --representations-file for analyze mode")
            return
        
        # Load representations
        extractor = RepresentationExtractor()
        representations = extractor.load_representations(args.representations_file)
        
        # Analyze
        analyzer = RepresentationAnalyzer(representations)
        
        # Train classifiers for all layers
        logger.info("Training harmful content classifiers...")
        classifier_results = analyzer.analyze_all_layers()
        
        # Analyze turn evolution
        logger.info("Analyzing turn-by-turn evolution...")
        evolution_results = {}
        for layer_name in analyzer.layer_data.keys():
            evolution_results[layer_name] = analyzer.analyze_turn_evolution(layer_name)
        
        # Save analysis results
        output_dir = Path(args.output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        analysis_file = output_dir / f"analysis_results_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump({
                'classifier_results': {k: {
                    'layer_name': v['layer_name'],
                    'accuracy': v['accuracy'],
                    'classification_report': v['classification_report'],
                    'test_samples': v['test_samples'],
                    'feature_dim': v['feature_dim']
                } for k, v in classifier_results.items()},
                'evolution_results': evolution_results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Analysis complete. Results saved to: {analysis_file}")
        
        # Print summary
        print("\n=== ANALYSIS SUMMARY ===")
        for layer_name, results in classifier_results.items():
            print(f"{layer_name}: {results['accuracy']:.3f} accuracy")


if __name__ == "__main__":
    main()
