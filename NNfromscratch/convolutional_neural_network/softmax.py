import numpy as np

class Softmax:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len  # divide by input length to reduce variance
        self.biases = np.zeros(nodes)

    def forward(self, input):
        """
        Returns a 1d numpy array containing the respective probability values.
        """

        self.last_input_shape = input.shape  # cache the inputs shape before we flatten it.

        input = input.flatten()
        self.last_input = input  # cache the input after we flatten it to 1-dimension

        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals  # cache the sum of inputs and weights + bias (value before softmax activation)

        exp = np.exp(totals)
        return exp / np.sum(exp)

    def backprop(self, d_L_d_out, learn_rate):

        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:  # only for the correct label d_L_d_out will be nonzero
                continue

            t_exp = np.exp(self.last_totals)

            S = np.sum(t_exp)  # sum of all e^probabilities

            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)  # gradient for other labels
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)  # gradient for correct label


            """
            now we back propagate just like in normal neural networks
            t = w * input + b
            """

            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            d_L_d_t = gradient * d_out_d_t


            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            # we need 2d arrays for @ but both matrices are 1D so we use np.newaxis to multiply dimensions (input_len, 1) & (1, nodes)
            # so final result will have shape (input_len, nodes) -> same as self.weights from our maxpool

            d_L_d_b = d_L_d_t * d_t_d_b

            d_L_d_inputs = d_t_d_inputs @ d_L_d_t
            # we multiply matrices with dimensions (input_len, nodes) and (nodes, 1) to get a result with length input_len.


            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
            return d_L_d_inputs.reshape(self.last_input_shape)  # reshape to undo our flattening during forward pass