#!/usr/bin/env python3
"""
Quick test script to verify the model setup is working correctly.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model_loading():
    """Test if we can load the DeepSeek-R1 model successfully."""
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    print("Testing model loading...")
    print(f"Model: {model_name}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✅ Tokenizer loaded successfully")
        
        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        print("✅ Model loaded successfully")
        
        # Test simple generation
        print("\nTesting simple generation...")
        test_prompt = "User: Hello, can you help me with a simple question?\n\nAssistant:"
        
        inputs = tokenizer(test_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        print(f"Test response: {response.strip()}")
        print("✅ Generation test successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run the setup test."""
    print("DeepSeek-R1 Model Setup Test")
    print("=" * 40)
    
    success = test_model_loading()
    
    if success:
        print("\n🎉 Setup test passed! You can now run the full inference script.")
        print("Run: python inference.py")
    else:
        print("\n❌ Setup test failed. Please check your environment and dependencies.")
        print("Make sure to install requirements: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
