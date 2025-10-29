#!/usr/bin/env python3
"""
Simplified activation steering for language models.

Usage:
    steerer = SteeringModel(
        model_path="path/to/model",
        steering_vectors={12: vector_tensor},  # or load from file
        alpha=1.0
    )
    
    # Single generation
    output = steerer.generate("What is 2+2?", max_tokens=512)
    
    # Batch generation
    outputs = steerer.generate_batch("What is 2+2?", n=10)
    
    steerer.close()
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union, List, Dict, Optional
import json


class SteeringModel:
    """Language model with activation steering."""
    
    def __init__(
        self,
        model_path: str,
        steering_vectors: Union[Dict[int, torch.Tensor], str] = None,
        alpha: float = 0.0,
        device: str = "auto"
    ):
        """
        Initialize steering model.
        
        Args:
            model_path: Path to model
            steering_vectors: Dict mapping layer_idx -> steering vector tensor,
                            or path to .pt file containing vectors
            alpha: Steering strength
            device: Device placement ("auto", "cuda", "cpu")
        """
        self.model_path = model_path
        self.alpha = alpha
        self.steering_vectors = {}
        
        # Load steering vectors
        if steering_vectors is not None:
            if isinstance(steering_vectors, str):
                self._load_vectors_from_file(steering_vectors)
            elif isinstance(steering_vectors, dict):
                self.steering_vectors = {k: v.float() for k, v in steering_vectors.items()}
            else:
                raise ValueError(f"steering_vectors must be dict or path, got {type(steering_vectors)}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on {self.device}")
        
        # Move steering vectors to device
        self._vector_cache = {}
        for layer_idx, vec in self.steering_vectors.items():
            self._vector_cache[layer_idx] = vec.to(
                device=self.device, 
                dtype=self.model.dtype
            ).view(1, 1, -1)
        
        # Register hooks
        self.hooks = []
        self._register_hooks()
        
        print(f"Steering on {len(self.steering_vectors)} layers with alpha={self.alpha}")
    
    def _load_vectors_from_file(self, path: str):
        """Load steering vectors from file."""
        path = Path(path)
        
        if path.suffix == '.pt':
            data = torch.load(path, map_location='cpu')
            
            if isinstance(data, torch.Tensor):
                if len(data.shape) == 2:
                    # Multi-layer: [num_layers, hidden_dim]
                    for i in range(data.shape[0]):
                        self.steering_vectors[i] = data[i].float()
                elif len(data.shape) == 1:
                    # Single vector - need layer specified
                    raise ValueError("Single vector .pt file requires specifying layer in dict format")
            elif isinstance(data, dict):
                self.steering_vectors = {k: v.float() for k, v in data.items()}
            else:
                raise ValueError(f"Unexpected .pt file format: {type(data)}")
                
        elif path.suffix == '.json':
            with open(path) as f:
                data = json.load(f)
            
            # Handle various JSON formats
            if isinstance(data, dict) and 'vectors' in data:
                for layer_str, vec_list in data['vectors'].items():
                    self.steering_vectors[int(layer_str)] = torch.tensor(vec_list, dtype=torch.float32)
            else:
                raise ValueError("JSON format not recognized. Expected {'vectors': {layer: [...]}}")
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        print(f"Loaded steering vectors for {len(self.steering_vectors)} layers")
    
    def _register_hooks(self):
        """Register forward hooks for steering."""
        for layer_idx in self.steering_vectors.keys():
            layer = self.model.model.layers[layer_idx]
            hook = self._make_hook(layer_idx)
            handle = layer.register_forward_hook(hook)
            self.hooks.append(handle)
    
    def _make_hook(self, layer_idx: int):
        """Create steering hook for a layer."""
        def hook_fn(module, input, output):
            # Extract hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Apply steering
            if self.alpha != 0:
                steering_vector = self._vector_cache[layer_idx]
                hidden_states = hidden_states + self.alpha * steering_vector
            
            # Return in original format
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states
        
        return hook_fn
    
    def set_alpha(self, alpha: float):
        """Update steering strength."""
        self.alpha = alpha
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
        return_full_text: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text with steering.
        
        Args:
            prompt: Input text
            max_tokens: Max new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample (False = greedy)
            return_full_text: If True, return prompt + generation; if False, only generation
            **kwargs: Additional generate() arguments
            
        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=temperature if do_sample else 1.0,
                    top_p=top_p if do_sample else 1.0,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                    **kwargs
                )
            
            # Decode immediately and move to CPU
            if return_full_text:
                text = self.tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
            else:
                # Only return generated portion
                generated_ids = outputs[0][input_ids.shape[1]:].cpu()
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        finally:
            # Explicitly free GPU memory
            del input_ids, attention_mask, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return text
    
    def generate_batch(
        self,
        prompts: Union[str, List[str]],
        n: int = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
        return_full_text: bool = False,
        batch_size: int = None,
        **kwargs
    ) -> List[str]:
        """
        Generate multiple completions.
        
        Args:
            prompts: Single prompt (will be repeated n times) or list of prompts
            n: Number of completions (only used if prompts is a string)
            max_tokens: Max new tokens per completion
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample
            return_full_text: If True, return prompt + generation; if False, only generation
            batch_size: Process in smaller batches to save memory (default: process all at once)
            **kwargs: Additional generate() arguments
            
        Returns:
            List of generated texts
        """
        # Handle input format
        if isinstance(prompts, str):
            if n is None:
                raise ValueError("Must specify n when prompts is a string")
            prompt_list = [prompts] * n
        else:
            prompt_list = prompts
        
        # Process in batches if requested
        if batch_size is None:
            batch_size = len(prompt_list)
        
        all_outputs = []
        
        for i in range(0, len(prompt_list), batch_size):
            batch_prompts = prompt_list[i:i+batch_size]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding="longest",
                    return_attention_mask=True,
                    padding_side="left" if hasattr(self.tokenizer, "padding_side") else None
                )
                self.tokenizer.padding_side = "left"
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_tokens,
                        temperature=temperature if do_sample else 1.0,
                        top_p=top_p if do_sample else 1.0,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        use_cache=True,
                        **kwargs
                    )
                
                # Decode each output and move to CPU immediately
                for j, output_ids in enumerate(outputs):
                    output_cpu = output_ids.cpu()
                    
                    if return_full_text:
                        text = self.tokenizer.decode(output_cpu, skip_special_tokens=True)
                    else:
                        # Find actual prompt length (excluding padding)
                        prompt_len = int(attention_mask[j].sum().item())
                        generated_ids = output_cpu[prompt_len:]
                        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    all_outputs.append(text)
                    del output_cpu
            finally:
                # Explicitly free GPU memory after each batch
                del input_ids, attention_mask, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return all_outputs
    
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def close(self):
        """Remove hooks and cleanup."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.clear_cache()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        if hasattr(self, 'hooks'):
            self.close()


# Utility functions

def load_steering_vectors(path: str) -> Dict[int, torch.Tensor]:
    """
    Load steering vectors from file.
    
    Args:
        path: Path to .pt or .json file
        
    Returns:
        Dict mapping layer index to steering vector
    """
    path = Path(path)
    vectors = {}
    
    if path.suffix == '.pt':
        data = torch.load(path, map_location='cpu')
        
        if isinstance(data, torch.Tensor):
            if len(data.shape) == 2:
                # Multi-layer format
                for i in range(data.shape[0]):
                    vectors[i] = data[i].float()
            elif len(data.shape) == 1:
                raise ValueError("Single vector needs layer specification")
        elif isinstance(data, dict):
            vectors = {k: v.float() for k, v in data.items()}
            
    elif path.suffix == '.json':
        with open(path) as f:
            data = json.load(f)
        if 'vectors' in data:
            for layer_str, vec_list in data['vectors'].items():
                vectors[int(layer_str)] = torch.tensor(vec_list, dtype=torch.float32)
    
    return vectors


def normalize_vectors(vectors: Dict[int, torch.Tensor], per_layer: bool = True) -> Dict[int, torch.Tensor]:
    """
    Normalize steering vectors.
    
    Args:
        vectors: Dict of steering vectors
        per_layer: If True, normalize each layer independently (recommended)
                  If False, normalize all vectors together (preserves relative magnitudes)
                  
    Returns:
        Normalized vectors
    """
    if per_layer:
        return {k: v / v.norm() for k, v in vectors.items()}
    else:
        # Stack all vectors and compute total norm
        all_vecs = torch.stack(list(vectors.values()))
        total_norm = all_vecs.norm()
        return {k: v / total_norm for k, v in vectors.items()}

