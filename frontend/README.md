# Probe Value Visualizer

Interactive web-based visualization of probe values across tokens.

## Setup

### Required Data Files

Place these files in the `frontend/` directory:

1. **`monitor.pt`** - List of tensors with probe values per token
2. **`prompttokens.pt`** (or `tokens.pt`) - List of lists of tokens (strings)
3. **`cheats.pt`** (or `cheating.pt`) - List of boolean values indicating if response is cheating

### Install Dependencies

```bash
pip install dash plotly torch
```

## Usage

```bash
cd frontend
python probe_visualizer.py
```

Then open http://localhost:8050 in your browser.

## Features

- **Dropdown**: Select which response to visualize
  - Shows if cheating or honest
  - Shows number of tokens
  
- **Line Plot**: Interactive plot of probe values vs token index
  - Hover to see exact values
  - Shows which token you're looking at
  
- **Token Display**: Visual representation of tokens
  - 🔵 Blue background = Low probe values
  - ⚪ White background = Mid probe values  
  - 🔴 Red background = High probe values
  - Hover over any token to see exact probe value
  
- **Stats Panel**: Shows mean, std, min, max probe values for selected response

## Data Preparation

If your tokens and cheating labels are in a different format, use this template:

```python
import torch

# Assuming you have:
# - probe_values: list of tensors (already in monitor.pt)
# - all_tokens: list of lists of token strings
# - is_cheating: list of booleans

# Save tokens
torch.save(all_tokens, 'frontend/tokens.pt')

# Save cheating labels  
torch.save(is_cheating, 'frontend/cheating.pt')
```

## Color Scale

Values are normalized globally across all responses for consistent comparison:
- Minimum probe value → Blue
- Middle probe value → White
- Maximum probe value → Red

## Tips

- Use the dropdown to quickly scan through responses
- Look for patterns: do cheating responses have different probe value distributions?
- The line plot helps identify which parts of the response trigger high/low probe values
- Hover over tokens to see exact values without cluttering the display

