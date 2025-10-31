#!/bin/bash
# Quick start script for probe visualizer

echo "🔍 Probe Value Visualizer"
echo "========================="
echo ""

# Check if required files exist
if [ ! -f "monitor.pt" ]; then
    echo "❌ ERROR: monitor.pt not found!"
    echo "   Please make sure monitor.pt is in the frontend/ directory"
    exit 1
fi

if [ ! -f "tokens.pt" ]; then
    echo "⚠️  WARNING: tokens.pt not found"
    echo "   Will use placeholder tokens"
fi

if [ ! -f "cheating.pt" ]; then
    echo "⚠️  WARNING: cheating.pt not found"
    echo "   Will use placeholder cheating labels"
fi

echo ""
echo "✓ Starting visualizer..."
echo ""
echo "Open your browser to: http://localhost:8050"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python probe_visualizer.py

