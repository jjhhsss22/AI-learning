import numpy as np
from numpy.random import randn

class RNN:
    def __init__(self, input_size, output_size, hidden_size=64):
        # Weights
        self.Whh = randn(hidden_size, hidden_size) / 1000  # divide by 0 to reduce value.
        self.Wxh = randn(hidden_size, input_size) / 1000   # not the best way but simple for this project
        self.Why = randn(output_size, hidden_size) / 1000

        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def feedforward(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))  # initialise our hidden state (vector of 0s)

        self.last_inputs = inputs
        self.last_hs = {0 : h}

        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)  # @ = matrix multiplication
            self.last_hs[i + 1] = h

        '''
        1. Multiply input vector x by input-to-hidden weights Wxh.
        2. Multiply previous hidden state h by hidden-to-hidden weights Whh.
        3. Add hidden bias bh.
        4. Apply tanh activation to get new hidden state h 
           and repeat until final hidden state h reached
        '''

        y = self.Why @ h + self.by  # calculate final output using hidden-to-output weight Why
                                    # and add output bias by
        return y, h

    def backprop(self, d_y, learn_rate = 2e-2):
        n = len(self.last_inputs)


        d_Why = d_y @ self.last_hs[n].T

        '''
        .T = transpose of vector in numpy
        d_y (dL/dy) has shape (output_size, 1) → column vector of output gradients.
        self.last_hs[n] has shape (hidden_size, 1) → column vector of the final hidden state.
        To form a weight gradient matrix (output_size, hidden_size), we need to find the transpose of hidden state t.
        '''

        d_by = d_y  # gradient for bias is just dL/dy


        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)  # initialise these gradients

        d_h = self.Why.T @ d_y  # different dL/dh for final h because there is no h after it.

        # BPTT (back propagation through time)

        for t in reversed(range(n)):
            temp = ((1 - self.last_hs[t + 1] ** 2) * d_h) # intermediate value: dL/dh * (1 - h^2)

            d_bh += temp  # dL/db = dL/dh * (1 - h^2)
            d_Whh += temp @ self.last_hs[t].T  # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
            d_Wxh += temp @ self.last_inputs[t].T  # dL/dWxh = dL/dh * (1 - h^2) * x

            d_h = self.Whh @ temp  # next dL/dh = dL/dh * (1 - h^2) * Whh


        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)
            # clipping to prevent gradient getting to extreme (exploding / vanishing gradient)
            # keep gradients between -1 & 1

        # gradient descent
        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by



