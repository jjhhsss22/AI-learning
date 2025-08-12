import numpy as np

# activation function - sigmoid
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# loss function - MSE
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()  # 1/n x sumof(y_true - y_pred)^2
                                          # where y_true & y_pred are both vectors with n elements



# neuron class
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, x):
        total = np.dot( self.weights, x) + self.bias  # dot product + bias value
        return sigmoid(total)

weights = np.array([0, 1])
bias = 4
neuron = Neuron(weights, bias)

x = np.array([2,3])  # inputs to the NN
print(neuron.feedforward(x))



# neural network class with
class NeuralNetwork:
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0  # each neuron has the same weights & biases

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias) # simple NN with two hidden neurons & one output neuron

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1

simple_network = NeuralNetwork()
x = np.array([2, 3])
print(simple_network.feedforward(x))



# gender prediction given height & weight

'''     Weight (-135)     Height(-66)     Gender
Alice        -2               -1             1
Bob          25                6             0
Charlie      17                4             0
Diana       -15               -6             1
'''

