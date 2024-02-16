import numpy as np
from mlxtend.data import loadlocal_mnist
import random

images_path = 'C:\\Users\\Rohit Negi\\Desktop\\digit_rec_scratch\\train-images.idx3-ubyte'
labels_path = 'C:\\Users\\Rohit Negi\\Desktop\\digit_rec_scratch\\train-labels.idx1-ubyte'

# Load the MNIST dataset
X, y = loadlocal_mnist(images_path=images_path, labels_path=labels_path)

# Set random seeds for reproducibility
np.random.seed(0)
random.seed(0)

# Splitting dataset
print("Shape of X:", X.shape) #Shape of X: (60000, 784)
print("Shape of y:", y.shape) #Shape of y: (60000,)

num_train = 50000
num_test = 10000
X_train = X[:num_train, :] / 255
y_train = np.zeros((num_train, 10))
y_train[np.arange(0, num_train), y[:num_train]] = 1
X_test = X[num_train:, :] / 255
y_test = np.zeros((num_test, 10))
y_test[np.arange(0, num_test), y[y.size - num_test:]] = 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / exps.sum()


def loss(predicted_output, desired_output):
    return 1 / 2 * np.sum((desired_output - predicted_output) ** 2)


class NeuralNetwork():
    def __init__(self, inputLayerNeuronsNumber, hiddenLayerNeuronsNumber1, hiddenLayerNeuronsNumber2,outputLayerNeuronsNumber):
        self.inputLayerNeuronsNumber = inputLayerNeuronsNumber
        self.hiddenLayerNeuronsNumber1 = hiddenLayerNeuronsNumber1
        self.hiddenLayerNeuronsNumber2 = hiddenLayerNeuronsNumber2
        self.outputLayerNeuronsNumber = outputLayerNeuronsNumber
        self.learning_rate = 0.01  # Lowered learning rate
        self.regularization = 0.0001  # Added L2 regularization
        self.hidden_weights1 = np.random.randn(hiddenLayerNeuronsNumber1, inputLayerNeuronsNumber) * np.sqrt( 2 / inputLayerNeuronsNumber)
        self.hidden_bias1 = np.zeros([hiddenLayerNeuronsNumber1, 1])
        self.hidden_weights2 = np.random.randn(hiddenLayerNeuronsNumber2, hiddenLayerNeuronsNumber1) * np.sqrt(2 / hiddenLayerNeuronsNumber1)
        self.hidden_bias2 = np.zeros([hiddenLayerNeuronsNumber2, 1])
        self.output_weights = np.random.randn(outputLayerNeuronsNumber, hiddenLayerNeuronsNumber2) * np.sqrt( 2 / hiddenLayerNeuronsNumber2)
        self.output_bias = np.zeros([outputLayerNeuronsNumber, 1])
        self.loss = []

    def train(self, inputs, desired_output):
        hidden_layer_in1 = np.dot(self.hidden_weights1, inputs) + self.hidden_bias1
        hidden_layer_out1 = sigmoid(hidden_layer_in1)

        hidden_layer_in2 = np.dot(self.hidden_weights2, hidden_layer_out1) + self.hidden_bias2
        hidden_layer_out2 = sigmoid(hidden_layer_in2)

        output_layer_in = np.dot(self.output_weights, hidden_layer_out2) + self.output_bias
        predicted_output = softmax(output_layer_in)

        error = desired_output - predicted_output

        d_predicted_output = error

        error_hidden_layer2 = np.dot(self.output_weights.T, d_predicted_output) * sigmoid_derivative(hidden_layer_out2)
        error_hidden_layer1 = np.dot(self.hidden_weights2.T, error_hidden_layer2) * sigmoid_derivative( hidden_layer_out1)

        # Update weights with L2 regularization
        self.output_weights += (np.dot(d_predicted_output, hidden_layer_out2.T) - 2 * self.regularization * self.output_weights) * self.learning_rate
        self.output_bias += d_predicted_output * self.learning_rate

        self.hidden_weights2 += (np.dot(error_hidden_layer2,hidden_layer_out1.T) - 2 * self.regularization * self.hidden_weights2) * self.learning_rate
        self.hidden_bias2 += error_hidden_layer2 * self.learning_rate

        self.hidden_weights1 += (np.dot(error_hidden_layer1,inputs.T) - 2 * self.regularization * self.hidden_weights1) * self.learning_rate
        self.hidden_bias1 += error_hidden_layer1 * self.learning_rate

        self.loss.append(loss(predicted_output, desired_output))

    def predict(self, inputs):
        hidden_layer_in1 = np.dot(self.hidden_weights1, inputs) + self.hidden_bias1
        hidden_layer_out1 = sigmoid(hidden_layer_in1)

        hidden_layer_in2 = np.dot(self.hidden_weights2, hidden_layer_out1) + self.hidden_bias2
        hidden_layer_out2 = sigmoid(hidden_layer_in2)

        output_layer_in = np.dot(self.output_weights, hidden_layer_out2) + self.output_bias
        predicted_output = softmax(output_layer_in)
        return predicted_output


# Training
nn = NeuralNetwork(784, 300, 150, 10)

num_epochs = 20
# Increased number of training epochs

for epoch in range(num_epochs):
    for i in range(X_train.shape[0]):
        inputs = np.array(X_train[i, :].reshape(-1, 1))
        desired_output = np.array(y_train[i, :].reshape(-1, 1))
        nn.train(inputs, desired_output)

    # Print the epoch number and loss
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {np.mean(nn.loss[-len(X_train):])}')

# Save the trained model
np.savez('neural_network_model.npz',
         hidden_weights1=nn.hidden_weights1,
         hidden_bias1=nn.hidden_bias1,
         hidden_weights2=nn.hidden_weights2,
         hidden_bias2=nn.hidden_bias2,
         output_weights=nn.output_weights,
         output_bias=nn.output_bias)

print("Model saved successfully.")

# Testing
prediction_list = []
for i in range(X_test.shape[0]):
    inputs = np.array(X_test[i].reshape(-1, 1))
    prediction_list.append(nn.predict(inputs))

correct_counter = 0
for i in range(len(prediction_list)):
    index = np.where(prediction_list[i] == np.amax(prediction_list[i]))[0][0]

    if y_test[i][index] == 1:
        correct_counter += 1

accuracy = correct_counter / num_test

print("Accuracy is : ", accuracy * 100, " %")
