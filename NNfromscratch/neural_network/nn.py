import numpy as np
import matplotlib.pyplot as plt

# activation function - sigmoid
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))  # this is derivative of sigmoid function

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

class NeuralNetworkForGender:
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)

        return  sum_h1, h1, sum_h2, h2, sum_o1, o1

    def train(self, data, all_y_trues):
        '''
        data is a (n x 2) numpy array, n = # of samples in the dataset.
        all_y_trues is a numpy array with n elements.
        Elements in all_y_trues correspond to those in data.
        '''

        learning_rate = 0.1
        epochs = 1000

        losses = []
        epoch_list = []

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1, h1, sum_h2, h2, sum_o1, o1 = self.feedforward(x)

                y_pred = o1


                #gradient descent
                # calculate partial derivatives using dLdw = dL/dpred * dpred/dh * dh/dw
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)


                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)


                # backpropagation
                # Neuron o1
                d_L_d_w6 = d_L_d_ypred * d_ypred_d_w6
                d_L_d_w5 = d_L_d_ypred * d_ypred_d_w5
                d_L_d_b3 = d_L_d_ypred * d_ypred_d_b3

                # Neuron h2
                d_L_d_w4 = d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                d_L_d_w3 = d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                d_L_d_b2 = d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron h1
                d_L_d_w2 = d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                d_L_d_w1 = d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                d_L_d_b1 = d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1


                self.w6 -= learning_rate * d_L_d_w6
                self.w5 -= learning_rate * d_L_d_w5
                self.b3 -= learning_rate * d_L_d_b3

                self.w4 -= learning_rate * d_L_d_w4
                self.w3 -= learning_rate * d_L_d_w3
                self.b2 -= learning_rate * d_L_d_b2

                self.w2 -= learning_rate * d_L_d_w2
                self.w1 -= learning_rate * d_L_d_w1
                self.b1 -= learning_rate * d_L_d_b1


                # loss per 10 epoches
                if epoch % 10 == 0:
                    y_preds = np.apply_along_axis(lambda row: self.feedforward(row)[-1], 1, data)
                    loss = mse_loss(all_y_trues, y_preds)
                    losses.append(loss)
                    epoch_list.append(epoch)
                    print("Epoch %d loss: %.3f" % (epoch, loss))


        # epoch vs loss graph
        plt.plot(epoch_list, losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Epochs vs losses")
        plt.show()



# dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])

# true results
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

network = NeuralNetworkForGender()
network.train(data, all_y_trues)