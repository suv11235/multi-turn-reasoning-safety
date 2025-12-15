import modal
import json

# Image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "numpy",
        "pandas",
        "scikit-learn",
        "h5py"
    )
)

local_dir = "/Users/suvajitmajumder/multi-turn-reasoning-safety"

# Add local files to image
image = image.add_local_dir(
    local_dir,
    remote_path="/workspace",
    copy=True,
    ignore=[".venv", ".git", "__pycache__", "representations", "wandb"]
)

# Create app with the fully configured image
app = modal.App("phase2-drift-experiment", image=image)

@app.function(
    gpu="A10G",
    timeout=1800  # 30 minutes
)
def run_drift_experiment():
    import os
    import sys
    os.chdir("/workspace")
    sys.path.append("/workspace")
    
    print("=" * 60)
    print("PHASE 2: Drift Analysis Experiment")
    print("=" * 60)
    
    # Step 1: Generate attacks with Gradual strategy
    print("\n[1/3] Generating Gradual Crescendo attacks...")
    from automated_crescendo import (
        AutomatedCrescendoPipeline, 
        HarmfulCategory, 
        EscalationStrategy
    )
    
    pipeline = AutomatedCrescendoPipeline(max_context_turns=3)
    
    results = pipeline.run_batch_testing(
        categories=[HarmfulCategory.VIOLENCE],
        strategies=[EscalationStrategy.GRADUAL],
        num_attempts=2
    )
    
    # Save results
    output_file = "gradual_attacks.json"
    pipeline.save_results(results, output_file)
    print(f"✓ Generated {len(results)} attacks, saved to {output_file}")
    
    # Print summary
    successful = [r for r in results if r.success]
    if len(results) > 0:
        print(f"  Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
    else:
        print(f"  No attacks generated - check template availability")
    
    # Step 2: Extract representations
    print("\n[2/3] Extracting representations from ALL attacks (successful + failed)...")
    from representation_extraction import RepresentationExtractor
    
    extractor = RepresentationExtractor(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        output_dir="representations_gradual"
    )
    
    # Extract from all attacks (limit to 2 for speed)
    all_reps = []
    # Use the first 2 results regardless of success
    targets = results[:2] 
    
    for attack_result in targets:
        print(f"  Processing attack: {attack_result.template_id} (Success: {attack_result.success})")
        reps = extractor.extract_representations_from_conversation(
            conversation_turns=attack_result.conversation_turns,
            conversation_id=attack_result.template_id,
            category=attack_result.category,
            strategy=attack_result.strategy,
            success=attack_result.success,
            is_harmful=False
        )
        all_reps.extend(reps)
    
    print(f"✓ Extracted {len(all_reps)} representation snapshots")
    
    # Step 3: Analyze drift
    print("\n[3/3] Analyzing drift patterns...")
    from representation_extraction import DriftAnalyzer
    
    analyzer = DriftAnalyzer(all_reps)
    
    # Get available layers
    layers = list(set([r.layer_name for r in all_reps]))
    print(f"  Analyzing {len(layers)} layers")
    
    # Analyze drift for middle layer
    if layers:
        target_layer = sorted(layers)[len(layers)//2]  # Pick middle layer
        print(f"\n  Target layer: {target_layer}")
        
        # Cosine drift
        cosine_metrics = analyzer.measure_cosine_drift(target_layer)
        print(f"\n  Cosine Drift Results:")
        if 'drift_data' in cosine_metrics:
            for conv_id, data in list(cosine_metrics['drift_data'].items())[:2]:
                print(f"    Conversation {conv_id}:")
                trajectory = data.get('cosine_trajectory', [])
                for turn, similarity in enumerate(trajectory, 1):
                    if isinstance(similarity, (int, float)):
                        print(f"      Turn {turn}: {similarity:.4f} similarity to initial")
                    else:
                        print(f"      Turn {turn}: {similarity} (non-numeric)")
        
        # Probe-based drift
        probe_metrics = analyzer.measure_drift(target_layer)
        print(f"\n  Probe-Based Drift Results:")
        if 'drift_data' in probe_metrics:
            for conv_id, trajectory in list(probe_metrics['drift_data'].items())[:2]:
                print(f"    Conversation {conv_id}:")
                for turn, prob in enumerate(trajectory, 1):
                    print(f"      Turn {turn}: {prob:.4f} harmful probability")
        elif 'error' in probe_metrics:
            print(f"    Warning: {probe_metrics['error']}")
    
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
    
    return {
        'num_attacks': len(results),
        'num_successful': len(successful),
        'num_representations': len(all_reps),
        'layers_analyzed': len(layers)
    }

if __name__ == "__main__":
    with app.run():
        result = run_drift_experiment.remote()
        print(f"\nFinal Summary: {result}")
