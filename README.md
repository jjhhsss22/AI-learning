# Neural Networks from Scratch

Core deep learning architectures implemented from first principles, to understand the mechanics rather than the API.

The NumPy implementations use no autograd: every forward pass, backward pass, gradient and update is written by hand. The Transformer is built in PyTorch from its component layers rather than with `nn.Transformer`.

## Contents

| Model | Framework | What's implemented | Location |
|---|---|---|---|
| Feed-forward network | NumPy | Neurons, sigmoid activation, MSE loss, backpropagation and SGD written by hand; trained on a small toy classification task, with a plot of the loss curve | `NNfromscratch/neural_network/` |
| Convolutional network | NumPy | 3×3 convolution layer, 2×2 max-pooling and softmax layers, each with its own forward and backward pass; trained on MNIST handwritten digits with cross-entropy loss | `NNfromscratch/convolutional_neural_network/` |
| Recurrent network | NumPy | Vanilla RNN with backpropagation through time; trained for sentiment classification on short text | `NNfromscratch/recurrent_neural_network/` |
| Transformer | PyTorch | Full encoder–decoder architecture: multi-head self-attention, positional embeddings, encoder and decoder blocks, and masking | `NNfromscratch/transformers/` |

## Running

```bash
pip install -r requirements.txt

# NumPy CNN on MNIST
cd NNfromscratch/convolutional_neural_network && python cnn.py

# NumPy RNN for sentiment
cd NNfromscratch/recurrent_neural_network && python rnn_main.py

# Feed-forward network
cd NNfromscratch/neural_network && python nn.py

# Transformer forward pass
cd NNfromscratch/transformers && python transformer.py
```

## What I took away

Notes drawn from the comments I wrote while building each model:

- **Max-pooling backprop is sparse.** The gradient only flows back to the position that held the max value in each 2×2 window. Every other position gets zero, because it didn't contribute to the output (`maxpool.py`).
- **RNN gradients need clipping.** Backpropagation through time multiplies by `Whh` at every step, so gradients can explode or vanish. I clip them to [-1, 1] to keep them from reaching extremes (`rnn.py`).
- **Why attention is scaled by √dₖ.** Dot products grow with dimension, which makes softmax "peaky" and its gradients vanish. Dividing by √dₖ keeps the scores in a usable range (`transformer.py`).
- **Masking happens before softmax.** Masked scores are set to −∞ so softmax turns them into exactly zero. The decoder ANDs a padding mask with a lower-triangular causal mask, so a token is only attended to if it isn't padding and isn't in the future (`transformer.py`).
- **Cross-attention doesn't literally reuse the encoder's keys and values.** The decoder derives its own K and V from the encoder's output through learned linear projections. "From the encoder" means the input, not the tensors themselves.

## References

- Vaswani et al., [*Attention Is All You Need*](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (NeurIPS 2017)
- Jay Alammar, [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/)
- [Transformer from scratch walkthrough](https://www.youtube.com/watch?v=U0s0f995w14) (video)

This is an ongoing project, and I add new architectures as I study them.
