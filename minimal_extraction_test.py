#!/usr/bin/env python3
"""
Minimal extraction test to validate the pipeline without full model loading.

This script tests the conversation processing and tokenization steps
without running the heavy model inference.
"""

import json
import torch
from transformers import AutoTokenizer
from automated_crescendo import HarmfulCategory, EscalationStrategy, AttackResult

def test_conversation_processing():
    """Test conversation context building and tokenization."""
    
    # Load tokenizer only (much lighter than full model)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load test attack data
    print("Loading attack data...")
    with open('test_attack_single.json', 'r') as f:
        attack_data = json.load(f)
    
    # Process first attack result
    data = attack_data[0]
    print(f"Processing: {data['template_id']}")
    print(f"Conversation turns: {len(data['conversation_turns'])}")
    
    def build_conversation_context(conversation_turns, current_question, k=3):
        """Build conversation context using sliding window approach."""
        # Select the k most recent conversation turns
        if len(conversation_turns) <= k:
            recent_turns = conversation_turns
        else:
            recent_turns = conversation_turns[-k:]
            
        # Build messages list for chat template
        messages = []
        for user_msg, assistant_msg in recent_turns:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        # Add the current user question
        messages.append({"role": "user", "content": current_question})
        
        # Format using the model's chat template
        context = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return context
    
    # Process each turn
    conversation_turns = data['conversation_turns']
    
    for turn_idx, (user_input, assistant_response) in enumerate(conversation_turns):
        print(f"\n=== Turn {turn_idx + 1} ===")
        print(f"User input: {user_input[:100]}...")
        print(f"Assistant response length: {len(assistant_response)} chars")
        
        # Build conversation context up to this turn
        context_turns = conversation_turns[:turn_idx]
        conversation_text = build_conversation_context(context_turns, user_input)
        
        print(f"Context length: {len(conversation_text)} chars")
        
        # Tokenize
        inputs = tokenizer(
            conversation_text,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
            padding=True
        )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        print(f"Token count: {input_ids.shape[1]}")
        print(f"Input shape: {input_ids.shape}")
        
        # Check for reasoning traces
        if '<think>' in assistant_response and '</think>' in assistant_response:
            start = assistant_response.find('<think>') + 7
            end = assistant_response.find('</think>')
            reasoning_trace = assistant_response[start:end].strip()
            print(f"Reasoning trace length: {len(reasoning_trace)} chars")
        else:
            print("No reasoning trace found")
    
    print(f"\n✅ Successfully processed {len(conversation_turns)} turns")
    print("Tokenization and context building working correctly!")
    
    return True

def test_memory_usage():
    """Test memory usage for different scenarios."""
    print("\n=== Memory Usage Test ===")
    
    # Get available memory info
    import psutil
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Memory before tokenizer: {memory_before:.1f} MB")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)
    
    memory_after_tokenizer = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory after tokenizer: {memory_after_tokenizer:.1f} MB")
    print(f"Tokenizer overhead: {memory_after_tokenizer - memory_before:.1f} MB")
    
    # Test tokenization of various lengths
    test_texts = [
        "Short text",
        "Medium length text " * 50,
        "Long text " * 200
    ]
    
    for i, text in enumerate(test_texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
        memory_current = process.memory_info().rss / 1024 / 1024
        print(f"Test {i+1} - Text length: {len(text)}, Tokens: {inputs['input_ids'].shape[1]}, Memory: {memory_current:.1f} MB")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing minimal extraction pipeline...")
    
    try:
        # Test conversation processing
        success1 = test_conversation_processing()
        
        # Test memory usage
        success2 = test_memory_usage()
        
        if success1 and success2:
            print("\n🎉 All tests passed! The extraction pipeline logic is working.")
            print("The issue is likely model loading memory requirements.")
        else:
            print("\n❌ Some tests failed.")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

