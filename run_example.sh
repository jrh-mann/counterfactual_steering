#!/bin/bash
# Example script to run steering experiments
# 
# Note: With the new design, you should just edit the variables
# in run_steering_experiment.py and run it directly:
#
#   python run_steering_experiment.py
#
# But this script shows how you could still run it from bash

echo "Running steering experiments..."
echo ""
echo "The script will sweep over multiple alpha values:"
echo "  - Alpha 0.0 (no steering)"
echo "  - Alpha 0.5 (low steering)"
echo "  - Alpha 1.0 (medium steering)"
echo "  - Alpha 1.5 (high steering)"
echo "  - Alpha 2.0 (very high steering)"
echo ""
echo "Results will be saved to results/ directory"
echo ""
echo "Press Ctrl+C to cancel, or wait 3 seconds to continue..."
sleep 3

python run_steering_experiment.py

echo ""
echo "All experiments complete! Check results/ directory for output."
echo ""
echo "To customize the experiments, edit the variables in main() function of run_steering_experiment.py"
