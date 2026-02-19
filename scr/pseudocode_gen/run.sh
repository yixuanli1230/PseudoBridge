#!/bin/bash
echo "🚀 Starting Pseudo-Code Generation Pipeline..."
# Ensure the current directory is in PYTHONPATH so the config module can be found
export PYTHONPATH=$PYTHONPATH:$(pwd)
python PseudoBridge/scr/pseudocode_gen/generator.py