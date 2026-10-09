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

- **Backpropagation is just the chain rule, applied carefully.** Writing gradients by hand for convolution and pooling made it clear why frameworks cache forward-pass values.
- **Why RNNs struggle with long sequences.** Backpropagation through time multiplies gradients across steps, so they vanish or explode, which is the problem attention removes.
- **Attention is a weighted lookup.** Queries score keys, softmax turns the scores into weights, and those weights mix the values. Masking stops the decoder from seeing future tokens.

## References

- Vaswani et al., [*Attention Is All You Need*](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (NeurIPS 2017)
- Jay Alammar, [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/)
- [Transformer from scratch walkthrough](https://www.youtube.com/watch?v=U0s0f995w14) (video)

This is an ongoing project, and I add new architectures as I study them.
