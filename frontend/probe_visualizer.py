#!/usr/bin/env python3
"""
Interactive web-based probe value visualizer.
Shows tokens colored by probe values with interactive hover.

Usage:
    python probe_visualizer.py
    
Then open http://localhost:8050 in your browser.
"""

import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, callback
import numpy as np
from pathlib import Path

# Initialize Dash app
app = Dash(__name__)

# Load data
print("Loading data...")
data_dir = Path(__file__).parent

# Load probe values
monitor_data = torch.load(data_dir / "monitor.pt")  # List of tensors
print(f"✓ Loaded {len(monitor_data)} responses with probe values")

# Try to load tokens and cheating labels
try:
    tokens_data = torch.load(data_dir / "prompttokens.pt")  # List of lists of tokens
    print(f"✓ Loaded tokens from prompttokens.pt")
except:
    try:
        tokens_data = torch.load(data_dir / "tokens.pt")  # Try alternate name
        print(f"✓ Loaded tokens from tokens.pt")
    except:
        print("⚠ prompttokens.pt or tokens.pt not found, will use placeholder tokens")
        tokens_data = [[f"token_{i}" for i in range(len(probe))] for probe in monitor_data]

try:
    cheating_labels = torch.load(data_dir / "cheats.pt")  # List of bool
    print(f"✓ Loaded cheating labels from cheats.pt")
except:
    try:
        cheating_labels = torch.load(data_dir / "cheating.pt")  # Try alternate name
        print(f"✓ Loaded cheating labels from cheating.pt")
    except:
        print("⚠ cheats.pt or cheating.pt not found, will use placeholder labels")
        cheating_labels = [False] * len(monitor_data)

# Normalize probe values globally for consistent color scale
all_probe_values = torch.cat([probe.flatten() for probe in monitor_data])
global_min = all_probe_values.min().item()
global_max = all_probe_values.max().item()
print(f"Probe value range: {global_min:.2f} to {global_max:.2f}")


def normalize_probe_values(probe_values):
    """Normalize probe values to 0-1 range for coloring."""
    return (probe_values - global_min) / (global_max - global_min)


def create_token_components(tokens, probe_values):
    """Create Dash components for tokens with background colors based on probe values."""
    normalized_values = normalize_probe_values(probe_values)
    
    # Use a diverging colorscale (blue=low, white=mid, red=high)
    token_components = []
    for token, raw_val, norm_val in zip(tokens, probe_values, normalized_values):
        # Map to color (using a red-white-blue scale)
        if norm_val < 0.5:
            # Blue to white
            intensity = norm_val * 2
            color = f"rgb({int(255*intensity)}, {int(255*intensity)}, 255)"
        else:
            # White to red
            intensity = (norm_val - 0.5) * 2
            color = f"rgb(255, {int(255*(1-intensity))}, {int(255*(1-intensity))})"
        
        # Create token span as a Dash component
        token_span = html.Span(
            token,
            style={
                'backgroundColor': color,
                'padding': '2px 4px',
                'margin': '1px',
                'display': 'inline-block',
                'borderRadius': '3px',
                'fontFamily': 'monospace',
                'cursor': 'pointer',
                'border': '1px solid #ddd'
            },
            title=f"Value: {raw_val:.2f}"
        )
        token_components.append(token_span)
        # Add space between tokens
        token_components.append(" ")
    
    return token_components


def create_probe_plot(probe_values, tokens):
    """Create line plot of probe values across tokens."""
    fig = go.Figure()
    
    # Create custom hover text with token, index, and value
    hover_text = [f"<b>Token {i}:</b> {token}<br><b>Value:</b> {val:.2f}" 
                  for i, (token, val) in enumerate(zip(tokens, probe_values.tolist()))]
    
    fig.add_trace(go.Scatter(
        x=list(range(len(probe_values))),
        y=probe_values.tolist(),
        mode='lines+markers',
        name='Probe Value',
        line=dict(color='rgb(31, 119, 180)', width=2),
        marker=dict(size=6),
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=hover_text,
    ))
    
    fig.update_layout(
        title="Probe Values Across Tokens",
        xaxis_title="Token Index",
        yaxis_title="Probe Value",
        hovermode='closest',
        height=400,
        template='plotly_white',
        font=dict(size=12)
    )
    
    return fig


# App layout
app.layout = html.Div([
    html.Div([
        html.H1("🔍 Probe Value Visualizer", style={'textAlign': 'center', 'color': '#2c3e50'}),
        html.P(
            f"Visualizing probe values for {len(monitor_data)} responses | "
            f"Range: {global_min:.1f} to {global_max:.1f}",
            style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px'}
        ),
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'}),
    
    html.Div([
        html.Label("Select Response:", style={'fontSize': '16px', 'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='response-dropdown',
            options=[
                {
                    'label': f"Response {i} - {'⚠️ CHEATING' if cheating_labels[i] else '✓ Honest'} ({len(tokens_data[i])} tokens)",
                    'value': i
                }
                for i in range(len(monitor_data))
            ],
            value=0,
            clearable=False,
            style={'width': '100%', 'fontSize': '14px'}
        ),
    ], style={'padding': '20px'}),
    
    html.Div([
        html.Div(id='response-info', style={
            'padding': '15px',
            'backgroundColor': '#fff',
            'borderRadius': '5px',
            'marginBottom': '20px',
            'border': '2px solid #3498db'
        }),
    ], style={'padding': '0 20px'}),
    
    html.Div([
        dcc.Graph(id='probe-plot'),
    ], style={'padding': '20px'}),
    
    html.Div([
        html.H3("Token Visualization", style={'color': '#2c3e50'}),
        html.P(
            "Hover over tokens to see exact probe values. "
            "🔵 Blue = Low values | ⚪ White = Mid values | 🔴 Red = High values",
            style={'color': '#7f8c8d', 'fontSize': '14px', 'marginBottom': '10px'}
        ),
        html.Div(id='token-display', style={
            'padding': '20px',
            'backgroundColor': '#fff',
            'borderRadius': '5px',
            'border': '1px solid #ddd',
            'lineHeight': '2.5',
            'fontSize': '13px',
            'maxHeight': '500px',
            'overflowY': 'auto'
        }),
    ], style={'padding': '20px'}),
    
    html.Div([
        html.P(
            "Color scale: Blue (low) → White (mid) → Red (high)",
            style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '12px', 'marginTop': '20px'}
        )
    ])
])


@callback(
    [Output('probe-plot', 'figure'),
     Output('token-display', 'children'),
     Output('response-info', 'children')],
    Input('response-dropdown', 'value')
)
def update_visualization(response_idx):
    """Update all visualizations when dropdown changes."""
    # Get data for selected response
    probe_values = monitor_data[response_idx]
    tokens = tokens_data[response_idx]
    is_cheating = cheating_labels[response_idx]
    
    # Create plot
    fig = create_probe_plot(probe_values, tokens)
    
    # Create token components
    token_display = html.Div(
        create_token_components(tokens, probe_values)
    )
    
    # Create info panel
    mean_val = probe_values.mean().item()
    std_val = probe_values.std().item()
    min_val = probe_values.min().item()
    max_val = probe_values.max().item()
    
    info_panel = html.Div([
        html.H3(
            f"{'⚠️ CHEATING DETECTED' if is_cheating else '✓ Honest Response'}", 
            style={
                'color': '#e74c3c' if is_cheating else '#27ae60',
                'marginBottom': '10px'
            }
        ),
        html.Div([
            html.Span(f"📊 Tokens: {len(tokens)} | ", style={'marginRight': '15px'}),
            html.Span(f"📈 Mean: {mean_val:.2f} | ", style={'marginRight': '15px'}),
            html.Span(f"📉 Std: {std_val:.2f} | ", style={'marginRight': '15px'}),
            html.Span(f"⬇️ Min: {min_val:.2f} | ", style={'marginRight': '15px'}),
            html.Span(f"⬆️ Max: {max_val:.2f}", style={'marginRight': '15px'}),
        ], style={'fontSize': '14px', 'color': '#34495e'})
    ])
    
    return fig, token_display, info_panel


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Starting Probe Visualizer")
    print("="*80)
    print("\nOpen your browser and go to: http://localhost:8050")
    print("\nPress Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=8050)

