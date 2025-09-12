#!/usr/bin/env python3
"""
Lightweight representation extraction using a smaller model.

This demonstrates the representation extraction pipeline with a smaller model
that can run on CPU without memory issues.
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightweightRepresentationExtractor:
    """Lightweight version using a smaller model for demonstration."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """Initialize with a lightweight model."""
        self.model_name = model_name
        self.device = torch.device("cpu")  # Force CPU for reliability
        
        logger.info(f"Loading lightweight model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Target the last layer of this smaller model
        self.target_layer = len(self.model.transformer.layer) - 1
        
        # Storage
        self.representations = []
        self.current_context = {}
        
        logger.info(f"Model loaded. Target layer: {self.target_layer}")
    
    def _register_hook(self):
        """Register hook on the target layer."""
        def hook_fn(module, input, output):
            # Store the output (handle tuple output)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.current_context['representation'] = hidden_states.detach().cpu().numpy()
        
        # Register on the last transformer layer
        layer = self.model.transformer.layer[self.target_layer]
        self.hook_handle = layer.register_forward_hook(hook_fn)
    
    def _remove_hook(self):
        """Remove the registered hook."""
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()
    
    def extract_conversation_representations(self, conversation_turns: List[Tuple[str, str]], 
                                           conversation_id: str, is_harmful: bool = False):
        """Extract representations from conversation turns."""
        logger.info(f"Processing conversation: {conversation_id}")
        
        # Register hook
        self._register_hook()
        
        try:
            for turn_idx, (user_input, assistant_response) in enumerate(conversation_turns):
                logger.info(f"  Turn {turn_idx + 1}: {len(user_input)} chars input")
                
                # For this lightweight demo, just process the user input
                # (since we don't have the complex chat template)
                inputs = self.tokenizer(
                    user_input,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,  # Smaller for lightweight model
                    padding=True
                )
                
                # Clear context
                self.current_context = {}
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Extract representation
                if 'representation' in self.current_context:
                    representation = self.current_context['representation']
                    
                    # Mean pool over sequence dimension
                    pooled_repr = np.mean(representation, axis=1)  # Shape: (1, hidden_dim)
                    
                    # Store with metadata
                    self.representations.append({
                        'conversation_id': conversation_id,
                        'turn_number': turn_idx + 1,
                        'layer_name': f"layer_{self.target_layer}",
                        'representation': pooled_repr,
                        'is_harmful': is_harmful,
                        'input_text': user_input,
                        'sequence_length': inputs['input_ids'].shape[1],
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    logger.info(f"    Captured representation: {pooled_repr.shape}")
        
        finally:
            self._remove_hook()
    
    def save_representations(self, filename: str = "lightweight_representations.json"):
        """Save representations as JSON (since they're small enough)."""
        logger.info(f"Saving {len(self.representations)} representations")
        
        # Convert numpy arrays to lists for JSON serialization
        json_data = []
        for rep in self.representations:
            json_rep = rep.copy()
            json_rep['representation'] = rep['representation'].tolist()
            json_data.append(json_rep)
        
        with open(filename, 'w') as f:
            json.dump({
                'representations': json_data,
                'total_count': len(json_data),
                'model_name': self.model_name,
                'target_layer': self.target_layer,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Saved to {filename}")
        return filename

def analyze_lightweight_representations(filename: str):
    """Analyze the extracted lightweight representations."""
    logger.info("Analyzing representations...")
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    representations = data['representations']
    
    # Separate harmful vs benign
    harmful_reps = [r for r in representations if r['is_harmful']]
    benign_reps = [r for r in representations if not r['is_harmful']]
    
    logger.info(f"Total representations: {len(representations)}")
    logger.info(f"Harmful: {len(harmful_reps)}, Benign: {len(benign_reps)}")
    
    if harmful_reps and benign_reps:
        # Simple analysis: compute mean representations
        harmful_features = np.array([r['representation'] for r in harmful_reps])
        benign_features = np.array([r['representation'] for r in benign_reps])
        
        harmful_mean = np.mean(harmful_features, axis=0)
        benign_mean = np.mean(benign_features, axis=0)
        
        # Compute distance between means
        distance = np.linalg.norm(harmful_mean - benign_mean)
        
        logger.info(f"Mean representation distance: {distance:.4f}")
        logger.info(f"Harmful mean magnitude: {np.linalg.norm(harmful_mean):.4f}")
        logger.info(f"Benign mean magnitude: {np.linalg.norm(benign_mean):.4f}")
        
        # Simple separability test
        if distance > 0.1:  # Arbitrary threshold
            logger.info("✅ Representations show potential separability!")
        else:
            logger.info("⚠️  Representations may not be well separated")
    
    return True

def main():
    """Main execution function."""
    logger.info("🧪 Lightweight Representation Extraction Demo")
    
    # Load attack data
    with open('test_attack_single.json', 'r') as f:
        attack_data = json.load(f)
    
    # Initialize extractor
    extractor = LightweightRepresentationExtractor()
    
    # Process the attack data (harmful)
    attack_result = attack_data[0]
    extractor.extract_conversation_representations(
        conversation_turns=attack_result['conversation_turns'],
        conversation_id=attack_result['template_id'],
        is_harmful=True
    )
    
    # Create some benign examples
    benign_conversations = [
        [("What is machine learning?", "Machine learning is a subset of AI..."), 
         ("Can you give an example?", "Sure, email spam detection is a common example...")],
        [("How do plants grow?", "Plants grow through photosynthesis..."),
         ("What do they need?", "They need sunlight, water, and nutrients...")]
    ]
    
    for i, conv in enumerate(benign_conversations):
        extractor.extract_conversation_representations(
            conversation_turns=conv,
            conversation_id=f"benign_{i:03d}",
            is_harmful=False
        )
    
    # Save representations
    output_file = extractor.save_representations()
    
    # Analyze results
    analyze_lightweight_representations(output_file)
    
    logger.info("✅ Lightweight extraction complete!")

if __name__ == "__main__":
    main()
