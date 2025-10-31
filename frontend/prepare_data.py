#!/usr/bin/env python3
"""
Helper script to prepare data for the probe visualizer.

If you have tokens and cheating labels stored somewhere else
(like in your experiment JSON files), use this script to convert them.
"""

import torch
import json
from pathlib import Path


def prepare_from_experiment_results(experiment_file, tokens_list, cheating_list, monitor_file):
    """
    Prepare data from experiment results.
    
    Args:
        experiment_file: Path to experiment JSON (has responses and judgments)
        tokens_list: List of lists of tokens (if already have it)
        cheating_list: List of booleans (if already have it)
        monitor_file: Path to monitor.pt file with probe values
    """
    # Load experiment results
    with open(experiment_file, 'r') as f:
        data = json.load(f)
    
    # Extract cheating labels from judgments
    judgments = data.get('judgments', [])
    cheating = [j.get('verdict') == 'reward_hacking' for j in judgments]
    
    print(f"Found {len(cheating)} responses")
    print(f"Cheating: {sum(cheating)}/{len(cheating)}")
    
    # If you have tokens, save them
    if tokens_list:
        torch.save(tokens_list, 'tokens.pt')
        print(f"✓ Saved tokens.pt")
    
    # Save cheating labels
    torch.save(cheating, 'cheating.pt')
    print(f"✓ Saved cheating.pt")
    
    # Verify monitor.pt exists
    if Path(monitor_file).exists():
        monitor = torch.load(monitor_file)
        print(f"✓ Found monitor.pt with {len(monitor)} responses")
        if len(monitor) != len(cheating):
            print(f"⚠️  WARNING: monitor.pt has {len(monitor)} responses but experiment has {len(cheating)}")
    else:
        print(f"⚠️  monitor.pt not found at {monitor_file}")


def prepare_from_variables(tokens_list, cheating_list, monitor_data):
    """
    Prepare data if you already have the variables in memory.
    
    Args:
        tokens_list: List of lists of tokens
        cheating_list: List of booleans
        monitor_data: List of tensors (probe values)
    """
    # Save everything
    torch.save(tokens_list, 'frontend/tokens.pt')
    torch.save(cheating_list, 'frontend/cheating.pt')
    torch.save(monitor_data, 'frontend/monitor.pt')
    
    print(f"✓ Saved {len(monitor_data)} responses to frontend/")
    print(f"  - tokens.pt: {len(tokens_list)} responses")
    print(f"  - cheating.pt: {sum(cheating_list)}/{len(cheating_list)} cheating")
    print(f"  - monitor.pt: {len(monitor_data)} probe value tensors")


def example_usage():
    """
    Example of how to use this script.
    """
    print("="*80)
    print("EXAMPLE USAGE")
    print("="*80)
    print("""
# If you have the data in variables:
tokens = [["The", "quick", "brown"], ["Another", "response"]]
cheating = [False, True]
monitor = [torch.randn(3), torch.randn(2)]

from prepare_data import prepare_from_variables
prepare_from_variables(tokens, cheating, monitor)

# Or if you have an experiment JSON:
from prepare_data import prepare_from_experiment_results
prepare_from_experiment_results(
    experiment_file='../results/experiment_alpha0p00_n48_20250101.json',
    tokens_list=your_tokens_list,
    cheating_list=None,  # Will extract from JSON
    monitor_file='monitor.pt'
)
    """)
    print("="*80)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage
        experiment_file = sys.argv[1]
        prepare_from_experiment_results(
            experiment_file=experiment_file,
            tokens_list=None,
            cheating_list=None,
            monitor_file='monitor.pt'
        )
    else:
        example_usage()
        print("\nTo use: python prepare_data.py <experiment_json_file>")
        print("Or import the functions in your notebook/script")

