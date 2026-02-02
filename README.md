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

##  How to Run
```bash
pip install -r requirements.txt
python demo.py

 Output

The demo prints tensor shapes confirming correct positional encoding behavior:

Learned Position Embedding Output Shape: torch.Size([2, 10, 512])
Sinusoidal Position Embedding Output Shape: torch.Size([2, 10, 512])

Key Takeaways

Transformers require position embeddings to model sequence order

Learnable embeddings are flexible but require training

Sinusoidal embeddings generalize to unseen sequence lengths

Both approaches produce compatible tensor shapes for Transformer layers


---

##  requirements.txt

```txt
torch

 learned_position_embedding.py
import torch
import torch.nn as nn

class LearnedPositionEmbedding(nn.Module):
    """
    Learnable position embeddings using neural network embeddings.
    """

    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int):
        super().__init__()

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Position embedding
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len = x.size()

        # Generate position indices
        position_ids = torch.arange(
            seq_len, device=x.device
        ).unsqueeze(0).expand(batch_size, seq_len)

        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(position_ids)

        return token_embeddings + position_embeddings

sinusoidal_position_embedding.py
import math
import torch
import torch.nn as nn

class FixedPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.
    """

    def __init__(self, embed_dim: int, max_seq_len: int):
        super().__init__()

        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )

        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register buffer (non-trainable)
        self.register_buffer("positional_encoding", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.positional_encoding[:, :seq_len, :]

demo.py
import torch
from learned_position_embedding import LearnedPositionEmbedding
from sinusoidal_position_embedding import FixedPositionalEncoding

VOCAB_SIZE = 1000
EMBED_DIM = 512
MAX_SEQ_LEN = 100
BATCH_SIZE = 2
SEQ_LEN = 10

# Dummy token input
dummy_tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

# Learned position embeddings
learned_pe = LearnedPositionEmbedding(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    max_seq_len=MAX_SEQ_LEN
)
learned_output = learned_pe(dummy_tokens)

# Sinusoidal position embeddings
fixed_pe = FixedPositionalEncoding(
    embed_dim=EMBED_DIM,
    max_seq_len=MAX_SEQ_LEN
)
dummy_embeddings = torch.zeros(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
fixed_output = dummy_embeddings + fixed_pe(dummy_embeddings)

print("Learned Position Embedding Output Shape:", learned_output.shape)
print("Sinusoidal Position Embedding Output Shape:", fixed_output.shape)

 outputs/sample_results.txt
Learned Position Embedding Output Shape: torch.Size([2, 10, 512])
Sinusoidal Position Embedding Output Shape: torch.Size([2, 10, 512])

--

Aram Radif


