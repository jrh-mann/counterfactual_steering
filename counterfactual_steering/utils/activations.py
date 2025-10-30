from tqdm import tqdm
import gc
import os
import torch
from concurrent.futures import ThreadPoolExecutor

def apply_chat_template(tokenizer, prompt, reasoning, completion):
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True
    ) + "<|channel|>analysis<|message|>" + \
    reasoning + "<|end|><|start|>assistant<|channel|>final<|message|>" + \
    completion + "<|return|>"

def get_start_of_sublist(tokenizer, prompt):
    tokens = tokenizer.tokenize(prompt)
    target = ['<|channel|>', 'analysis', '<|message|>']
    for i in range(len(tokens) - len(target) + 1):
        if tokens[i:i+len(target)] == target:
            return i + 3
    raise ValueError("Not found")

def store_activations(model, prompts, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(prompts, output_dir + "/prompts.pt")

    with torch.no_grad():
        for index, prompt in enumerate(tqdm(prompts)):
            residual_stream = []
            #start_of_response = get_start_of_sublist(model.tokenizer, prompt)
            with model.trace(prompt) as gen:
                for layer in model.model.layers:
                    residual_stream.append(layer.output.save()[0])
            acts = torch.stack(residual_stream)
            #indexed_acts = acts[:,start_of_response:]
            cpuacts = acts.cpu()
            torch.save(cpuacts, output_dir + f"/{index}.pt")

            del residual_stream
            del acts
            del indexed_acts
            torch.cuda.empty_cache()

def load_activations(output_dir, max_workers=8):
    acts = [act for act in os.listdir(output_dir) if act != "prompts.pt"]
    
    def load_single_file(filename):
        return torch.load(os.path.join(output_dir, filename))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use list() to force execution and maintain order
        results = list(tqdm(
            executor.map(load_single_file, acts),
            total=len(acts),
            desc="Loading activations"
        ))
    
    return results