# Transformer Hangman Solver

A neural network-based solution to the classic game of Hangman using a Transformer architecture. This project trains a transformer model to predict missing letters in partially revealed words, achieving competitive performance across various word lengths.

## Overview

This project implements a transformer-based approach to solve Hangman by:
1. Training a transformer model on millions of masked word examples
2. Using the model to predict the most likely letters for each position
3. Implementing an intelligent guessing strategy that considers letter probabilities across all masked positions

## Architecture

The model uses a **Transformer Encoder** architecture with the following specifications:
- **Model Size**: 25.2M parameters
- **Embedding Dimension**: 512
- **Number of Heads**: 8
- **Number of Layers**: 4
- **Vocabulary Size**: 29 characters (a-z, underscore, period, hyphen)
- **Max Sequence Length**: 25 characters

### Key Components

- **Character Embeddings**: Each character is embedded into a 512-dimensional space
- **Positional Encoding**: Learned positional embeddings for sequence position
- **Transformer Encoder**: Multi-head self-attention layers for contextual understanding
- **Output Layer**: Linear projection to vocabulary size for character prediction

## Training Process

### Dataset Generation
The training dataset is created by:
1. Loading 224,377 words from the English dictionary
2. Generating all possible masked combinations for each word (2^n combinations for n unique characters)
3. Creating 20M training examples and 1.4M validation examples
4. Converting to integer sequences for model training

### Training Details
- **Training Examples**: 20,000,000
- **Validation Examples**: 1,400,000
- **Batch Size**: 4,096
- **Learning Rate**: 3e-4 with cosine annealing scheduler
- **Optimizer**: AdamW with weight decay 0.0
- **Loss Function**: Cross-Entropy Loss
- **Training Time**: ~6 minutes on H100 GPU

## Performance Metrics

### Overall Performance
- **Training Loss**: 0.1686
- **Validation Loss**: 0.3047
- **In-Sample Win Rate**: **71.70%**
- **Out-of-Sample Win Rate**: **69.80%**

### Key Observations
- **Strong generalization**: 71.70% in-sample vs 69.80% out-of-sample performance
- **Consistent performance**: Model maintains high win rates across different word sets
- **Improved training**: Lower training loss and higher win rates indicate better model convergence

## How It Works

### Inference Process
1. **Input**: Partially revealed word with underscores for missing letters
2. **Model Prediction**: Transformer outputs probability distributions for each position
3. **Ensemble Strategy**: Averages probabilities across all masked positions
4. **Letter Selection**: Chooses the most likely unguessed letter
5. **Iteration**: Continues until word is solved or max guesses (6) reached

### Example Game
```
Word: "reperplex"
Masked: "_________"
Guesses: ['e', 'r', 'd', 'v', 'a', 't', 'l', 'p', 'x']
Result: SOLVED in 9 guesses
```

## Files

- `hangman.ipynb`: Main training and evaluation notebook
- `hangman.py`: Game implementation and baseline solver
- `model.pth`: Trained transformer model (25.2M parameters)
- `training_dictionary.txt`: 224,377 English words for training
- `test_dictionary.txt`: Test words for evaluation
- `generate_dictionaries.py`: Script to generate training data
- `test.py`: Utility functions for testing
- `train_dataset.pkl`: Pre-processed training dataset (2.0GB)
- `val_dataset.pkl`: Pre-processed validation dataset (148MB)
- `pyproject.toml`: Project configuration and dependencies
- `uv.lock`: Lock file for reproducible dependency management

## Requirements

- **Python**: >=3.12
- **PyTorch**: >=2.7.1
- **NumPy**: <2
- **Matplotlib**: >=3.10.3
- **tqdm**: >=4.67.1
- **pandas**: >=2.3.0
- **Jupyter**: >=1.1.1

## Setup and Usage

### Installation
This project uses `uv` for dependency management. To set up:

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

### Usage

1. **Training**: Run the `hangman.ipynb` notebook to train the model
2. **Inference**: Use the trained model for interactive hangman solving
3. **Evaluation**: Test performance on custom word lists

The model demonstrates that transformer architectures can effectively learn character-level patterns in English words and apply this knowledge to solve word-guessing games with competitive performance.