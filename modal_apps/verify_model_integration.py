import modal
import torch
import numpy as np

# Image with dependencies
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
    copy=True,  # Copy files into image layer at build time
    ignore=[".venv", ".git", "__pycache__", "representations"]
)

# Create app with the fully configured image
app = modal.App("phase2-integration-test", image=image)

@app.function(
    gpu="A10G", # Downgrade from H100 for 1.5B model
    timeout=600
)
def test_model_integration():
    import os
    import sys
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Contents of /: {os.listdir('/')}")
    print(f"Contents of /: {os.listdir('/')}")
    # print(f"Contents of /root: {os.listdir('/root')}")

    if os.path.exists("/workspace"):
        print(f"Contents of /workspace: {os.listdir('/workspace')}")
        os.chdir("/workspace")
        sys.path.append("/workspace") # Ensure it's in python path
    else:
        print("WARNING: /workspace not found!")
        # Fallback to verify logic
        
    from steering_vectors import CircuitBreaker
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Model loaded successfully.")
    
    # Test Circuit Breaker Registration
    # 1. Create dummy vector for a valid layer
    # DeepSeek-R1-Distill-Qwen-1.5B is Qwen2 based.
    # Architecture usually: model.layers.X
    
    layer_name = "model.layers.10" # Pick a middle layer
    hidden_dim = model.config.hidden_size
    
    # Random steering vector
    steering_vec = torch.randn(hidden_dim)
    vectors = {layer_name: steering_vec}
    
    print(f"Testing CircuitBreaker on {layer_name}...")
    cb = CircuitBreaker(model, vectors, alpha=2.0)
    
    # 2. Run forward pass WITHOUT breaker
    input_text = "Hello, world!"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out_clean = model(**inputs).logits
        
    # 3. Run forward pass WITH breaker
    cb.register()
    with torch.no_grad():
        out_steered = model(**inputs).logits
    cb.remove()
    
    # 4. Compare
    diff = (out_clean - out_steered).abs().mean().item()
    print(f"Logits difference (Mean Abs): {diff}")
    
    if diff < 1e-5:
        raise Exception("Circuit Breaker had no effect on output! Hooks might be failing.")
    
    print("Integration test passed: Circuit Breaker effectively modified forward pass.")

if __name__ == "__main__":
    with app.run():
        test_model_integration.remote()
