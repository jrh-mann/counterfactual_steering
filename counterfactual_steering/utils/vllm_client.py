"""Minimal client for batch querying vLLM server."""

from typing import List, Optional, Dict, Any
from openai import OpenAI, AsyncOpenAI
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class VLLMClient:
    """Client for interacting with vLLM server via OpenAI-compatible API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        timeout: float = 60.0
    ):
        """
        Initialize vLLM client.
        
        Args:
            base_url: Base URL of the vLLM server (default: http://localhost:8000/v1)
            api_key: API key (not needed for local vLLM, but required by OpenAI client)
            timeout: Request timeout in seconds
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout
        )
        self._cached_model = None
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from the server."""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Warning: Could not fetch models from server: {e}")
            return []
    
    def get_default_model(self) -> Optional[str]:
        """Get the default model from the server (first available model)."""
        if self._cached_model is None:
            models = self.get_available_models()
            if models:
                self._cached_model = models[0]
            else:
                # Fallback to a common default
                self._cached_model = "microsoft/phi-2"
        return self._cached_model
    
    def generate(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        max_workers: int = 10,
        return_full_response: bool = False,
        prefill: Optional[str] = None,
        **kwargs
    ) -> List[Any]:
        """
        Generate completions for a batch of prompts using parallel processing.
        
        Args:
            prompts: List of input prompts
            model: Model name (if None, will auto-detect from server)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_workers: Maximum number of parallel workers
            return_full_response: If True, returns full response dict with all metadata
            prefill: Optional text to prefill the assistant's response (model continues from here)
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            List of generated text completions (str) or full response dicts if return_full_response=True
        """
        if not prompts:
            return []
        
        # Auto-detect model if not provided
        if model is None:
            model = self.get_default_model()
        
        def _generate_single(prompt: str) -> Any:
            """Generate completion for a single prompt."""
            # Build messages array
            messages = [{"role": "user", "content": prompt}]
            
            # Add assistant prefill if provided
            if prefill:
                messages.append({"role": "assistant", "content": prefill})
            
            request_params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            response = self.client.chat.completions.create(**request_params)
            
            if return_full_response:
                # Return complete response as dict
                completion_text = response.choices[0].message.content or ""
                prefill_text = prefill if prefill else ""
                
                result = {
                    "prompt": prompt,
                    "prefill": prefill if prefill else None,
                    "completion": completion_text,
                    "full_text": prefill_text + completion_text,
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                    }
                }
                
                # Include additional fields if present (e.g., reasoning tokens, logprobs)
                if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                    result["logprobs"] = response.choices[0].logprobs
                
                # For models that support reasoning/thinking (like o1)
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    result["reasoning"] = response.choices[0].message.reasoning_content
                
                # Include the raw response object for full access
                result["raw_response"] = response.model_dump() if hasattr(response, 'model_dump') else None
                
                return result
            else:
                # Return full text (prefill + completion) if prefill was used
                completion_text = response.choices[0].message.content or ""
                if prefill:
                    return prefill + completion_text
                return completion_text
        
        # Process prompts in parallel using ThreadPoolExecutor
        responses = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_generate_single, prompt): idx 
                for idx, prompt in enumerate(prompts)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    responses[idx] = future.result()
                except Exception as e:
                    responses[idx] = f"Error: {str(e)}"
        
        return responses
    
    def generate_batch(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        max_workers: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate completions for a batch of prompts with full response details using parallel processing.
        
        Args:
            prompts: List of input prompts
            model: Model name (if None, will auto-detect from server)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_workers: Maximum number of parallel workers
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            List of response dictionaries with full details
        """
        if not prompts:
            return []
        
        # Auto-detect model if not provided
        if model is None:
            model = self.get_default_model()
        
        def _generate_single_detailed(prompt: str) -> Dict[str, Any]:
            """Generate completion with full details for a single prompt."""
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            result = {
                "prompt": prompt,
                "completion": response.choices[0].message.content or "",
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }
            }
            
            # Include additional fields if present
            if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                result["logprobs"] = response.choices[0].logprobs
            
            # For models that support reasoning/thinking
            if hasattr(response.choices[0].message, 'reasoning_content'):
                result["reasoning"] = response.choices[0].message.reasoning_content
            
            # Include the raw response object for complete access
            result["raw_response"] = response.model_dump() if hasattr(response, 'model_dump') else None
            
            return result
        
        # Process prompts in parallel using ThreadPoolExecutor
        responses = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_generate_single_detailed, prompt): idx 
                for idx, prompt in enumerate(prompts)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    responses[idx] = future.result()
                except Exception as e:
                    responses[idx] = {
                        "prompt": prompts[idx],
                        "completion": f"Error: {str(e)}",
                        "model": model,
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    }
        
        return responses
    
    def generate_with_messages(
        self,
        messages_list: List[List[Dict[str, str]]],
        model: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        max_workers: int = 10,
        return_full_response: bool = False,
        **kwargs
    ) -> List[Any]:
        """
        Generate completions using custom message arrays for full control.
        This allows multi-turn conversations, system prompts, and prefilling.
        
        Args:
            messages_list: List of message arrays, where each array contains dicts with 'role' and 'content'
                          Example: [[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]]
            model: Model name (if None, will auto-detect from server)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_workers: Maximum number of parallel workers
            return_full_response: If True, returns full response dict with all metadata
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            List of generated text completions (str) or full response dicts if return_full_response=True
            
        Example:
            # Prefill assistant response
            messages = [
                {"role": "user", "content": "Write a story about a cat"},
                {"role": "assistant", "content": "Once upon a time, there was a cat named"}
            ]
            results = client.generate_with_messages([messages], max_tokens=100)
        """
        if not messages_list:
            return []
        
        # Auto-detect model if not provided
        if model is None:
            model = self.get_default_model()
        
        def _generate_single(messages: List[Dict[str, str]]) -> Any:
            """Generate completion for a single message array."""
            request_params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            response = self.client.chat.completions.create(**request_params)
            
            if return_full_response:
                # Extract prefill if the last message was from assistant
                prefill = messages[-1]["content"] if messages and messages[-1]["role"] == "assistant" else None
                completion_text = response.choices[0].message.content or ""
                prefill_text = prefill if prefill else ""
                
                result = {
                    "messages": messages,
                    "prefill": prefill,
                    "completion": completion_text,
                    "full_text": prefill_text + completion_text,
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                    }
                }
                
                # Include additional fields if present
                if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                    result["logprobs"] = response.choices[0].logprobs
                
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    result["reasoning"] = response.choices[0].message.reasoning_content
                
                result["raw_response"] = response.model_dump() if hasattr(response, 'model_dump') else None
                
                return result
            else:
                return response.choices[0].message.content or ""
        
        # Process messages in parallel using ThreadPoolExecutor
        responses = [None] * len(messages_list)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_generate_single, messages): idx 
                for idx, messages in enumerate(messages_list)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    responses[idx] = future.result()
                except Exception as e:
                    responses[idx] = f"Error: {str(e)}"
        
        return responses
    
    async def generate_async(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """
        Generate completions for a batch of prompts using async processing (faster for many requests).
        
        Args:
            prompts: List of input prompts
            model: Model name (if None, will auto-detect from server)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            List of generated text completions
        """
        if not prompts:
            return []
        
        # Auto-detect model if not provided
        if model is None:
            model = self.get_default_model()
        
        async_client = AsyncOpenAI(
            base_url=self.client.base_url,
            api_key=self.client.api_key,
            timeout=self.client.timeout
        )
        
        async def _generate_single_async(prompt: str) -> str:
            """Generate completion for a single prompt asynchronously."""
            response = await async_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content
        
        # Process all prompts concurrently
        tasks = [_generate_single_async(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        results = []
        for resp in responses:
            if isinstance(resp, Exception):
                results.append(f"Error: {str(resp)}")
            else:
                results.append(resp)
        
        await async_client.close()
        return results


def format_prompt(task: str, rules: str, grader: str) -> str:
    """Format a reward hacking prompt for the model."""
    prompt = f"{task}\n\n{rules}\n\n{grader}"
    return prompt


def format_prompts_batch(
    tasks: List[str],
    rules: List[str],
    graders: List[str]
) -> List[str]:
    """
    Efficiently format a batch of reward hacking prompts.
    
    Args:
        tasks: List of task descriptions
        rules: List of rule descriptions
        graders: List of grader code blocks
        
    Returns:
        List of formatted prompts
    """
    if not (len(tasks) == len(rules) == len(graders)):
        raise ValueError("All input lists must have the same length")
    
    # Use list comprehension for efficient batch formatting
    prompts = [
        f"{task}\n\n{rule}\n\n{grader}"
        for task, rule, grader in zip(tasks, rules, graders)
    ]
    
    return prompts


def format_prompts_from_dicts(prompt_dicts: List[Dict[str, str]]) -> List[str]:
    """
    Format prompts from a list of dictionaries containing 'task', 'rules', and 'grader' keys.
    
    Args:
        prompt_dicts: List of dictionaries with 'task', 'rules', and 'grader' keys
        
    Returns:
        List of formatted prompts
    """
    tasks = [d["task"] for d in prompt_dicts]
    rules = [d["rules"] for d in prompt_dicts]
    graders = [d["grader"] for d in prompt_dicts]
    
    return format_prompts_batch(tasks, rules, graders)


if __name__ == "__main__":
    # Example usage
    client = VLLMClient()
    
    # Batch query example
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "What is 2+2?"
    ]
    
    print("Generating completions with parallel processing...")
    start_time = time.time()
    completions = client.generate(prompts, max_tokens=50, temperature=0.7, max_workers=3)
    elapsed = time.time() - start_time
    
    print(f"\nProcessed {len(prompts)} prompts in {elapsed:.2f} seconds")
    for prompt, completion in zip(prompts, completions):
        print(f"\nPrompt: {prompt}")
        print(f"Completion: {completion}")
    
    # Example with format_prompts_batch
    print("\n" + "="*50)
    print("Example with format_prompts_batch:")
    
    tasks = ["Task 1", "Task 2"]
    rules = ["Rule 1", "Rule 2"]
    graders = ["Grader 1", "Grader 2"]
    
    formatted = format_prompts_batch(tasks, rules, graders)
    print(f"Formatted {len(formatted)} prompts")

