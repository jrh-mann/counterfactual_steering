# Counterfactual Steering Experiment - Usage Guide

## Quick Start

1. **Edit the configuration in `run_steering_experiment.py`**:

```python
def main():
    # Edit these variables:
    alphas = [0.0, 0.5, 1.0, 1.5, 2.0]  # Alpha values to sweep
    n_runs = 48                          # Responses per alpha
    batch_size = 12                      # Batch size
    max_tokens = 2560                    # Max tokens
    
    # Update API key and paths if needed
    api_key = "your-api-key-here"
    steering_vec_path = 'counterfactual_steering/directions/meandiffalllayers.pt'
    problem_file = 'counterfactual_steering/data/rebenchprompt.txt'
```

2. **Run the script**:

```bash
python run_steering_experiment.py
```

That's it! No command-line arguments needed.

## What It Does

The script will:
1. Load the model and steering vectors once
2. Run experiments for each alpha value in the list
3. Generate `n_runs` responses for each alpha
4. Judge each response for reward hacking
5. Save individual results to `results/experiment_alpha{X}_n{Y}_{timestamp}.json`
6. Print a summary comparing all alpha values

## Example Output

```
================================================================================
LOADING RESOURCES
================================================================================
Loading steering vectors...
✓ Loaded steering vectors
Loading problem text...
✓ Loaded problem text
Initializing model...
✓ Model loaded

================================================================================
RUNNING EXPERIMENTS - 5 alpha value(s)
================================================================================

================================================================================
EXPERIMENT 1/5: Alpha = 0.0
================================================================================
Generating 48 responses in 4 batch(es) of up to 12...
Batch 1/4: Generating 12 responses... ✓ (12/48 total)
  → Judging batch 1... ✓
...

================================================================================
GENERATING PLOT
================================================================================
✓ Plot saved to results/rh_vs_alpha_n48_20250101_123456.png

================================================================================
SUMMARY OF ALL EXPERIMENTS
================================================================================

Alpha 0.00:
  Reward Hacking: 12/48 (25.0%)
  Honest: 36/48 (75.0%)
  Unclear: 0/48
  Errors: 0/48

Alpha 1.00:
  Reward Hacking: 5/48 (10.4%)
  Honest: 43/48 (89.6%)
  Unclear: 0/48
  Errors: 0/48

...
```

## Configuration Options

All configuration is in the `main()` function:

### Alpha Sweep
```python
# Sweep over multiple values
alphas = [0.0, 0.5, 1.0, 1.5, 2.0]

# Or single experiment
alphas = [0.0]

# Fine-grained sweep
alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0]
```

### Experiment Size
```python
n_runs = 48          # Total responses per alpha (default: 48)
batch_size = 12      # Responses per batch (default: 12)
max_tokens = 2560    # Max tokens per response (default: 2560)
max_retries = 3      # Retries for errors (default: 3)
```

### Paths
```python
steering_vec_path = 'counterfactual_steering/directions/meandiffalllayers.pt'
problem_file = 'counterfactual_steering/data/rebenchprompt.txt'
output_dir = 'results'  # Where to save results
```

### API
```python
api_key = "sk-or-v1-..."
model_name = 'openai/gpt-oss-20b'
```

## Output Files

### JSON Results
Each experiment generates a JSON file:
- `results/experiment_alpha0p00_n48_20250101_123456.json`
- `results/experiment_alpha1p00_n48_20250101_123458.json`
- etc.

Each file contains:
- `experiment_config`: Configuration used
- `stats`: Summary statistics
- `judgments`: Individual verdicts
- `responses`: Full text of responses

### Visualization
When running multiple alpha values, an automatic plot is generated:
- `results/rh_vs_alpha_n48_20250101_123456.png`

The plot shows:
- **Red line**: Reward Hacking percentage vs steering strength
- **Green line**: Honest optimization percentage vs steering strength
- **Title**: Includes sample size (n=X per α)
- **Labels**: Each data point is labeled with its percentage
- **Grid**: For easy reading of values

## Tips

- **Quick test**: Set `n_runs = 12` and `alphas = [0.0]` for a quick test
- **Memory**: Reduce `batch_size` to 8 or 6 if you have memory issues
- **Speed**: The script automatically retries errors and runs batches efficiently
- **Comparison**: Run multiple alphas in one go to compare steering effects
- **Plotting**: Plot is only generated when `len(alphas) > 1` - single experiments won't create a plot
- **Visualization**: The plot is saved as PNG and automatically displayed if your environment supports it

## Benefits

- ✅ No command-line arguments to remember
- ✅ Easy alpha sweeps
- ✅ All configuration in one place
- ✅ Model loaded once for all experiments
- ✅ Automatic summary comparison
- ✅ No asyncio/notebook issues
