#!/bin/bash

# Check if image path is provided
if [ -z "$1" ]; then
    echo "Usage: ./run_model.sh <path_to_image>"
    exit 1
fi

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the model
python run_model.py "$1"

# Deactivate the virtual environment
deactivate
