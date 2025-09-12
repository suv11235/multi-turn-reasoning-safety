#!/usr/bin/env python3
"""
Analyze real DeepSeek-R1 representations from harmful vs benign conversations.

This script combines and analyzes the representations we extracted from 
actual DeepSeek-R1 model inference.
"""

import h5py
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from datetime import datetime

def load_representations(filepath):
    """Load representations from HDF5 file."""
    representations = []
    
    with h5py.File(filepath, 'r') as f:
        for group_name in f.keys():
            group = f[group_name]
            
            # Load representation array
            representation = group["representation"][:]
            
            # Load metadata
            metadata = dict(group.attrs)
            
            # Mean pool over sequence dimension for analysis
            if len(representation.shape) == 3:  # (batch, seq, hidden)
                pooled_repr = np.mean(representation, axis=(0, 1))  # (hidden,)
            else:
                pooled_repr = representation.flatten()
            
            representations.append({
                'representation': pooled_repr,
                'conversation_id': metadata['conversation_id'],
                'turn_number': metadata['turn_number'],
                'layer_name': metadata['layer_name'],
                'is_harmful': metadata['is_harmful'],
                'category': metadata.get('category', None),
                'strategy': metadata.get('strategy', None),
                'sequence_length': representation.shape[1] if len(representation.shape) > 1 else 1,
                'timestamp': metadata['timestamp']
            })
    
    return representations

def analyze_deepseek_representations():
    """Main analysis of real DeepSeek representations."""
    print("🧠 Analyzing Real DeepSeek-R1 Representations")
    print("=" * 50)
    
    # Load harmful representations
    print("Loading harmful representations...")
    harmful_file = "real_deepseek_representations/representations_20250908_095238.h5"
    harmful_reps = load_representations(harmful_file)
    
    # Load benign representations  
    print("Loading benign representations...")
    benign_file = "real_deepseek_representations/representations_20250908_100033.h5"
    benign_reps = load_representations(benign_file)
    
    # Combine datasets
    all_reps = harmful_reps + benign_reps
    
    print(f"\n📊 Dataset Summary:")
    print(f"Total representations: {len(all_reps)}")
    print(f"Harmful: {len(harmful_reps)}")
    print(f"Benign: {len(benign_reps)}")
    print(f"Feature dimension: {all_reps[0]['representation'].shape[0]}")
    
    # Analyze sequence length evolution
    print(f"\n📏 Sequence Length Analysis:")
    for rep in all_reps:
        label = "HARMFUL" if rep['is_harmful'] else "BENIGN"
        print(f"  {rep['conversation_id']} Turn {rep['turn_number']}: {rep['sequence_length']} tokens ({label})")
    
    # Prepare data for classification
    X = np.array([rep['representation'] for rep in all_reps])
    y = np.array([1 if rep['is_harmful'] else 0 for rep in all_reps])
    
    print(f"\n🧮 Statistical Analysis:")
    print(f"Representation shape: {X.shape}")
    print(f"Label distribution: Harmful={np.sum(y)}, Benign={len(y) - np.sum(y)}")
    
    # Compute basic statistics
    harmful_features = X[y == 1]
    benign_features = X[y == 0]
    
    if len(harmful_features) > 0 and len(benign_features) > 0:
        harmful_mean = np.mean(harmful_features, axis=0)
        benign_mean = np.mean(benign_features, axis=0)
        
        # Distance between means
        mean_distance = np.linalg.norm(harmful_mean - benign_mean)
        
        # Magnitude analysis
        harmful_magnitude = np.mean([np.linalg.norm(rep) for rep in harmful_features])
        benign_magnitude = np.mean([np.linalg.norm(rep) for rep in benign_features])
        
        print(f"\n📐 Representation Geometry:")
        print(f"Mean representation distance: {mean_distance:.6f}")
        print(f"Harmful mean magnitude: {harmful_magnitude:.6f}")
        print(f"Benign mean magnitude: {benign_magnitude:.6f}")
        print(f"Magnitude ratio (harmful/benign): {harmful_magnitude/benign_magnitude:.6f}")
        
        # Dimensionality analysis
        harmful_std = np.std(harmful_features, axis=0)
        benign_std = np.std(benign_features, axis=0)
        
        # Find dimensions with largest differences
        diff_vector = np.abs(harmful_mean - benign_mean)
        top_diff_dims = np.argsort(diff_vector)[-10:]  # Top 10 most different dimensions
        
        print(f"\n🎯 Top Discriminative Dimensions:")
        for i, dim in enumerate(reversed(top_diff_dims)):
            diff = diff_vector[dim]
            h_val = harmful_mean[dim]
            b_val = benign_mean[dim]
            print(f"  Dim {dim:4d}: diff={diff:.6f}, harmful={h_val:.6f}, benign={b_val:.6f}")
    
    # Classification analysis
    if len(set(y)) > 1:  # Need both classes
        print(f"\n🤖 Classification Analysis:")
        
        # Simple classification (if we have enough data)
        if len(X) >= 4:  # Minimum for train/test split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.4, random_state=42, stratify=y
                )
                
                # Train classifier
                clf = LogisticRegression(random_state=42, max_iter=1000)
                clf.fit(X_train, y_train)
                
                # Evaluate
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                print(f"Train samples: {len(X_train)} (Harmful: {sum(y_train)}, Benign: {len(y_train)-sum(y_train)})")
                print(f"Test samples: {len(X_test)} (Harmful: {sum(y_test)}, Benign: {len(y_test)-sum(y_test)})")
                print(f"Classification accuracy: {accuracy:.4f}")
                
                if len(X_test) > 0:
                    report = classification_report(y_test, y_pred, output_dict=True)
                    print(f"Precision (harmful): {report['1']['precision']:.4f}")
                    print(f"Recall (harmful): {report['1']['recall']:.4f}")
                    print(f"F1-score (harmful): {report['1']['f1-score']:.4f}")
                
            except ValueError as e:
                print(f"Could not perform train/test split: {e}")
                print("Dataset too small for reliable classification")
        else:
            print("Dataset too small for classification analysis")
    
    # Turn-by-turn analysis
    print(f"\n🔄 Turn-by-Turn Evolution:")
    
    # Group by conversation
    conversations = {}
    for rep in all_reps:
        conv_id = rep['conversation_id']
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(rep)
    
    for conv_id, conv_reps in conversations.items():
        conv_reps.sort(key=lambda x: x['turn_number'])
        is_harmful = conv_reps[0]['is_harmful']
        label = "HARMFUL" if is_harmful else "BENIGN"
        
        print(f"\n  {conv_id} ({label}):")
        
        for i, rep in enumerate(conv_reps):
            magnitude = np.linalg.norm(rep['representation'])
            print(f"    Turn {rep['turn_number']}: magnitude={magnitude:.6f}, tokens={rep['sequence_length']}")
            
            if i > 0:
                # Compute distance from previous turn
                prev_rep = conv_reps[i-1]['representation']
                curr_rep = rep['representation']
                distance = np.linalg.norm(curr_rep - prev_rep)
                print(f"      → Distance from turn {rep['turn_number']-1}: {distance:.6f}")
    
    # Save analysis results
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_summary': {
            'total_representations': len(all_reps),
            'harmful_count': len(harmful_reps),
            'benign_count': len(benign_reps),
            'feature_dimension': all_reps[0]['representation'].shape[0]
        },
        'statistical_analysis': {
            'mean_distance': float(mean_distance) if 'mean_distance' in locals() else None,
            'harmful_magnitude': float(harmful_magnitude) if 'harmful_magnitude' in locals() else None,
            'benign_magnitude': float(benign_magnitude) if 'benign_magnitude' in locals() else None
        }
    }
    
    with open('real_deepseek_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Analysis complete! Results saved to real_deepseek_analysis_results.json")
    
    return results

if __name__ == "__main__":
    analyze_deepseek_representations()

