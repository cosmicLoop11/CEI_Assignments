# Text Generation using RNN, LSTM and GRU

## Overview

This assignment implements and compares three popular Recurrent Neural Network architectures for text generation:

- Vanilla RNN (SimpleRNN)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

The models are trained on a custom text corpus and learn sequential patterns, grammar, and contextual relationships between words to generate meaningful text based on a given seed phrase.

---

## Objective

The objective of this project is to understand how different recurrent neural network architectures learn textual patterns and generate new text sequences.

The project demonstrates:

- Text preprocessing and tokenization
- Sequence generation using n-grams
- Padding and input preparation
- Training deep learning sequence models
- Comparing training behavior of RNN, LSTM, and GRU
- Generating text from trained models

---

## Features

- Custom text corpus
- Word-level tokenization
- Sliding window (n-gram) sequence generation
- Sequence padding using Keras
- Three separate deep learning architectures:
  - SimpleRNN
  - LSTM
  - GRU
- Loss comparison visualization
- Text generation using next-word prediction
- Training over 200 epochs
- Enhanced embedding dimensions and hidden layers

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## Project Workflow

### 1. Data Preparation

- Input text corpus is cleaned and processed.
- Tokenizer converts words into integer indices.

### 2. Sequence Creation

Example:

Input sentence:

```
elephants are highly social animals
```

Generated sequences:

```
elephants are
elephants are highly
elephants are highly social
elephants are highly social animals
```

---

### 3. Model Building

Three architectures are implemented:

#### Vanilla RNN

- Embedding Layer
- SimpleRNN Layer
- Dense Output Layer

#### LSTM

- Embedding Layer
- LSTM Layer
- Dense Output Layer

#### GRU

- Embedding Layer
- GRU Layer
- Dense Output Layer

---

### 4. Model Training

All models are trained using:

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Epochs: 200

---

### 5. Text Generation

The trained models generate text using:

```python
np.argmax()
```

to select the most probable next word.

---

## Results

### Training Loss Comparison

The project visualizes training loss across 200 epochs for:

- Vanilla RNN
- LSTM
- GRU

This helps compare learning behavior and convergence speed among the three architectures.

---

### Sample Generated Text

**Seed Phrase**

```text
their
```

### Observation

Although all three models were trained on the same corpus and configuration, they learned different contextual relationships:

- RNN focused on memory and migration behavior.
- LSTM generated information related to elephant diet.
- GRU generated text about elephant trunk functionality.

This demonstrates how different recurrent architectures can capture different sequence patterns within the same dataset.

## Learning Outcomes

Through this assignment, I learned:

- Sequence modeling using Deep Learning
- Recurrent Neural Networks
- LSTM gates and memory mechanisms
- GRU architecture and optimization
- Text preprocessing for NLP tasks
- Language generation techniques
- Comparing neural network architectures
