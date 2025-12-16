import numpy as np
import random

from rnn import RNN
from data import train_data, test_data

# create a list of unique vocabs in the train data
vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
print ('Vocab size:', vocab_size)

# assign indices to each unique vocab
word_to_index = { w : i for i, w in enumerate(vocab) }
idx_to_word = { i : w for i, w in enumerate(vocab) }


def createInputs(text):  # Returns an array of one-hot vectors representing the words
                         # from the input string
    inputs = []

    for w in text.split(' '):
        v = np.zeros((vocab_size, 1))  # creates a column vector of zeros with length = vocab_size
        v[word_to_index[w]] = 1  # sets the value at the position of the current word = 1
        inputs.append(v)  # append the one-hot vectors
    return inputs

def softmax(xs):
    return np.exp(xs) / np.sum(np.exp(xs))  # softmax activation function


rnn = RNN(vocab_size, 2)

inputs = createInputs('i am very good')
output, h = rnn.feedforward(inputs)
probabilities = softmax(output)  # returns two probabilities (positive, negative) that sums up to 1

def processData(data, backprop=True):
    """
    Returns the RNN's loss and accuracy for the given data.
    """

    items = list(data.items())
    random.shuffle(items)

    loss = 0
    num_correct = 0

    # Loop over each training example
    for x, y in items:
        inputs = createInputs(x)
        target = int(y)

        out, _ = rnn.feedforward(inputs)
        probs = softmax(out)  # list / matrix of probabilities (in this case [positive, negative])

        loss -= np.log(probs[target].item())
        num_correct += int(np.argmax(probs) == target)

        if backprop:
            # Build dL/dy
            d_L_d_y = probs
            d_L_d_y[target] -= 1

            '''
            when calculating dL/dy,
            Start with d_L_d_y = probs (i.e. gradient = predicted probabilities).
            Subtract 1 from the probability of the correct class (target). Given we are using softmax + cross-entropy loss
            
            Example: If probs = [0.15, 0.02, 0.10, 0.70, 0.03] and target = 3:
                      d_L_d_y = [0.15, 0.02, 0.10, -0.30, 0.03]
            '''

            # Backward propagation
            rnn.backprop(d_L_d_y)

    return loss / len(data), num_correct / len(data)


# training our RNN model
for epoch in range(1000):
  train_loss, train_acc = processData(train_data)

  if epoch % 100 == 99:
    print('--- Epoch %d' % (epoch + 1))
    print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

    test_loss, test_acc = processData(test_data, backprop=False)
    print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))
