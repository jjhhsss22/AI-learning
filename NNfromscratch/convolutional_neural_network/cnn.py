import numpy as np
from tensorflow.keras.datasets import mnist
from convlayer import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

# Load MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalise pixel values to be between 0-1
train_images = train_images[:1000]
train_labels = train_labels[:1000]

test_images = test_images[:1000]
test_labels = test_labels[:1000]

conv = Conv3x3(8)  # converts to (26, 26, 8)
pool = MaxPool2()  # converts to (13, 13, 8)
softmax = Softmax(13 * 13 * 8, 10)  # flattens to (1352,) and converts to (10,)


def forward(image, label):
    """
    Completes a forward pass of the CNN
    and calculates the accuracy and cross-entropy loss (L = -ln(pc) where pc = predicted probability for class c).
    - image is a 2d numpy array
    - label is a digit
    """

    out = conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

def train(im, label, learning_rate=.005):

    out, loss, acc = forward(im, label)

    # initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    gradient = softmax.backprop(gradient, learning_rate)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, learning_rate)

    return loss, acc


# training
for epoch in range(3):  # 3 epochs
    print('--- Epoch %d ---' % (epoch + 1))

    # shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    loss = 0
    num_correct = 0

    for i, (image, label) in enumerate(zip(train_images, train_labels)):

        if i > 0 and i % 100 == 99:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' % (i + 1, loss / 100, num_correct))
            loss = 0
            num_correct = 0

        l, acc = train(image, label)
        loss += l
        num_correct += acc


# testing
loss = 0
num_correct = 0

for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)
