#!/bin/bash

# Default values
LEARNING_RATE=0.0001
EPOCHS=30
BATCH_SIZE=20
DROPOUT_RATE=0.5

# Parse command line arguments
while getopts l:e:b:d: flag
do
    case "${flag}" in
        l) LEARNING_RATE=${OPTARG};;
        e) EPOCHS=${OPTARG};;
        b) BATCH_SIZE=${OPTARG};;
        d) DROPOUT_RATE=${OPTARG};;
    esac
done

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Update config.json with new parameters
jq --argjson lr $LEARNING_RATE --argjson epochs $EPOCHS --argjson batch $BATCH_SIZE --argjson dropout $DROPOUT_RATE '
    .learning_rate = $lr |
    .epochs = $epochs |
    .batch_size = $batch |
    .dropout_rate = $dropout
' config.json > temp.json && mv temp.json config.json

# Run the training script
python train_model.py
