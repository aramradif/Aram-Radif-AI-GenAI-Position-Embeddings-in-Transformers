# Aram-Radif-AI-GenAI-Position-Embeddings-in-Transformers

Transformers lack sequence awareness due to parallel attention. Position embeddings inject order information either through learnable embeddings or fixed sinusoidal encodings. Learnable embeddings offer flexibility, while sinusoidal embeddings provide extrapolation to unseen sequence lengths with no additional parameters.

Repository Structure
position-embeddings-transformers/
│
├── README.md
├── requirements.txt
├── learned_position_embedding.py
├── sinusoidal_position_embedding.py
├── demo.py
└── outputs/
    └── sample_results.txt

# Position Embeddings in Transformers (PyTorch)

Transformers process tokens **in parallel**, which means sequence order is not inherently preserved.
To inject information about token order, **position embeddings** are added to token embeddings.

This repository implements and compares two widely used approaches:
1. Learnable (Neural Network) Position Embeddings
2. Fixed (Sinusoidal) Position Embeddings

##Rationale

Unlike RNNs, Transformers:
- Do not process tokens sequentially
- Are permutation-invariant by default

Position embeddings allow the model to understand:
> Which token comes before or after another token

## Implemented Methods

### Method 1 — Learnable Position Embeddings
- Uses trainable embedding layers
- Learns position representations during training
- Used in models like **BERT**

### Method 2 — Fixed (Sinusoidal) Position Embeddings
- Uses sine and cosine functions
- No learnable parameters
- Used in the original **Transformer (Vaswani et al.)**

##  Mathematical Formulation

### Even Dimensions
\[
PE_{(pos, i)} = \sin\left( pos \cdot e^{-\frac{\log(10000)}{d_{model}} \cdot i} \right)
\]

### Odd Dimensions
\[
PE_{(pos, i)} = \cos\left( pos \cdot e^{-\frac{\log(10000)}{d_{model}} \cdot (i - 1)} \right)
\]
The constant `log(10000)` ensures stability across short- and long-range dependencies.
--
Aram Radif

