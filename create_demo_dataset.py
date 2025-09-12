#!/usr/bin/env python3
"""
Create a synthetic demonstration dataset for representation analysis.

This script creates mock representation data to demonstrate the analysis pipeline
while we work on optimizing the actual extraction process.
"""

import numpy as np
import json
from datetime import datetime
from representation_extraction import (
    RepresentationExtractor, 
    RepresentationData, 
    HarmfulCategory, 
    EscalationStrategy
)

def create_synthetic_representations():
    """Create synthetic representation data for testing."""
    representations = []
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create representations for different layers
    layers = ["model.layers.0", "model.layers.7", "model.layers.14"]
    
    # Generate benign conversations (from our actual baseline)
    for conv_idx in range(5):
        for turn in range(1, 4):  # 3 turns per conversation
            for layer in layers:
                # Create representation with some structure
                # Benign representations have smaller magnitude in later layers
                layer_depth = int(layer.split('.')[-1])
                magnitude = 1.0 - (layer_depth / 28.0) * 0.3  # Decrease with depth
                
                representation = np.random.normal(0, magnitude, (1, 12, 1536))
                
                rep = RepresentationData(
                    conversation_id=f"benign_{conv_idx:03d}",
                    turn_number=turn,
                    layer_name=layer,
                    representation=representation,
                    is_harmful=False,
                    timestamp=datetime.now().isoformat()
                )
                representations.append(rep)
    
    # Generate harmful conversations (synthetic crescendo attacks)
    harmful_categories = [HarmfulCategory.VIOLENCE, HarmfulCategory.DRUGS, HarmfulCategory.WEAPONS]
    strategies = [EscalationStrategy.GRADUAL, EscalationStrategy.CONTEXTUAL]
    
    for conv_idx in range(5):
        category = harmful_categories[conv_idx % len(harmful_categories)]
        strategy = strategies[conv_idx % len(strategies)]
        success = conv_idx % 2 == 0  # 50% success rate
        
        for turn in range(1, 4):  # 3 turns per conversation
            for layer in layers:
                # Harmful representations have different patterns
                layer_depth = int(layer.split('.')[-1])
                
                if layer_depth <= 7:  # Early layers
                    # Early layers show less discrimination
                    representation = np.random.normal(0, 0.8, (1, 12, 1536))
                else:  # Later layers
                    # Later layers show more distinct patterns for harmful content
                    if turn >= 2:  # Escalation turns
                        # Add structure that represents escalation
                        base = np.random.normal(0, 0.7, (1, 12, 1536))
                        # Add escalation signal in specific dimensions
                        base[0, :, :100] += np.random.normal(0.5, 0.2, (12, 100))
                        representation = base
                    else:
                        representation = np.random.normal(0, 0.8, (1, 12, 1536))
                
                rep = RepresentationData(
                    conversation_id=f"harmful_{conv_idx:03d}_{category.value}",
                    turn_number=turn,
                    layer_name=layer,
                    representation=representation,
                    is_harmful=True,
                    category=category,
                    strategy=strategy,
                    success=success,
                    timestamp=datetime.now().isoformat()
                )
                representations.append(rep)
    
    return representations

def main():
    print("Creating synthetic demonstration dataset...")
    
    # Create synthetic representations
    representations = create_synthetic_representations()
    
    print(f"Created {len(representations)} synthetic representations")
    print(f"Benign: {sum(1 for r in representations if not r.is_harmful)}")
    print(f"Harmful: {sum(1 for r in representations if r.is_harmful)}")
    print(f"Layers: {list(set(r.layer_name for r in representations))}")
    
    # Save using the representation extractor
    extractor = RepresentationExtractor(output_dir="demo_representations")
    extractor.representations = representations
    
    output_file = extractor.save_representations("synthetic_demo.h5")
    print(f"Saved synthetic dataset to: {output_file}")
    
    # Print summary statistics
    print("\n=== Dataset Summary ===")
    for layer in ["model.layers.0", "model.layers.7", "model.layers.14"]:
        layer_reps = [r for r in representations if r.layer_name == layer]
        harmful_count = sum(1 for r in layer_reps if r.is_harmful)
        benign_count = len(layer_reps) - harmful_count
        print(f"{layer}: {harmful_count} harmful, {benign_count} benign")

if __name__ == "__main__":
    main()
