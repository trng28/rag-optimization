import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

class LLMWrapper:
    """Wrapper for LLM with activation extraction"""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_length: int = 512
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        # Storage cho activations
        self.activations = {}
        self.hooks = []
    
    def register_activation_hooks(self):
        """Register hooks để collect activations"""
        def get_activation(name):
            def hook(module, input, output):
                # Store hidden states
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
            return hook
        
        # Register for layers
        for name, module in self.model.named_modules():
            if 'layer' in name or 'block' in name:
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        collect_activations: bool = False
    ) -> Dict:
        """
        Generate response from prompt
        
        Returns:
            Dict 'response', 'activations' (if collect_activations=True)
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        ).to(self.device)
        
        if collect_activations:
            self.register_activation_hooks()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        result = {'response': response.strip()}
        
        if collect_activations:
            # Extract last token activations from each layer
            activations_list = []
            for name in sorted(self.activations.keys()):
                act = self.activations[name]
                # Get last token: [batch, seq_len, hidden] -> [batch, hidden]
                last_token_act = act[:, -1, :].cpu()
                activations_list.append(last_token_act)
            
            # Stack: [num_layers, hidden_dim]
            if activations_list:
                result['activations'] = torch.cat(activations_list, dim=0)
            
            self.remove_hooks()
        
        return result
    
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Dict]:
        """Batch generation"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results