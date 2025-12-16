"""VLLM-accelerated inference wrapper."""
from inference import MultiTurnJailbreakTester
from transformers import AutoTokenizer

class VLLMJailbreakTester(MultiTurnJailbreakTester):
    def __init__(self, model_name: str, max_context_turns: int = 3):
        # Do NOT call super().__init__ as it loads the HF model
        # We manually initialize what we need
        self.model_name = model_name
        self.max_context_turns = max_context_turns
        self.device = "cuda"
        
        print(f"Loading vLLM model: {model_name}")
        from vllm import LLM, SamplingParams
        
        # Initialize vLLM
        # tensor_parallel_size=1 since we are typically on a single A10G/A100 in Modal
        self.llm = LLM(
            model=model_name, 
            trust_remote_code=True, 
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.90 # Leave some overhead (e.g. for vllm internal state)
        )
        
        # Default sampling params
        self.sampling_params = SamplingParams(
            temperature=0.7, 
            top_p=0.9, 
            max_tokens=2048 # Increased specifically for R1 reasoning traces
        )
        
        # Initialize tokenizer independently for context building logic
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_batch(self, prompts):
        """Generate responses for a batch of prompts."""
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        # Clean and return responses
        responses = []
        for output in outputs:
            text = output.outputs[0].text
            responses.append(self.clean_response(text))
        return responses
    
    # We reuse clean_response, build_conversation_context etc. from parent class 
    # because they are pure logic methods that don't depend on self.model
