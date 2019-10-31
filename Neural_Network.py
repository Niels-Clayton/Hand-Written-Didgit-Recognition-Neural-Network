import numpy as np
import scipy.special as sci


class NeuralNet:
    def __init__(self, training_set, answer_set):

        self.inputs = training_set
        self.answers = answer_set
        self.lr = 0.1

        self.input_size = training_set.shape[1]
        self.hidden_layer_size = (28*28)
        self.output_layer_size = 10

        self.hidden_weights_1 = 2 * np.random.random((self.input_size, self.hidden_layer_size)) - 1
        self.output_weights = 2 * np.random.random((self.hidden_layer_size, self.output_layer_size)) - 1

        self.correct_output = None
        self.total_error = None

        self.layer_1_output = None
        self.layer_1_error = None
        self.layer_1_delta = None

        self.output = None
        self.output_error = None
        self.output_delta = None


    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            # return (x*(1-x))
            return 1/(1+(sci.expit(-x))) * (1 - 1/(1+(sci.expit(-x))))
        return 1/(1+(sci.expit(-x)))

    def forward_propagate(self, X):
        self.layer_1_output = self.sigmoid(np.dot(X, self.hidden_weights_1))
        self.output = self.sigmoid(np.dot(self.layer_1_output, self.output_weights))

    def calculate_error(self):
        self.output_error = self.correct_output - self.output
        self.total_error = np.mean(np.abs(self.output_error))

    def back_propagate(self, input, answer):

        self.correct_output = answer
        self.forward_propagate(input.copy())
        self.calculate_error()

        # calculate output delta
        self.output_delta = self.output_error * self.sigmoid(self.output, derivative=True)

        # Calculate hidden layer 1 error
        self.layer_1_error = np.dot(self.output_delta, np.transpose(self.output_weights))

        # calculate hidden layer 1 delta
        self.layer_1_delta = self.layer_1_error * self.sigmoid(self.layer_1_output, derivative=True)

        # update weights using calculated deltas
        self.output_weights += self.lr * np.dot(np.transpose(self.layer_1_output), self.output_delta)
        self.hidden_weights_1 += self.lr * np.dot(np.transpose(input), self.layer_1_delta)


    def train(self):
        pos = np.random.randint(self.inputs.shape[0] - 10)
        self.back_propagate(self.inputs[pos:pos + 10], self.answers[pos:pos + 10])
        count = 0

        while self.total_error > 1e-5:
            if count % 1000 == 0:
                print(self.answers[pos:pos+3])
                print(self.output[0:3])
                print()
                count = 0
            pos = np.random.randint(self.inputs.shape[0]-10)
            self.back_propagate(self.inputs[pos:pos+10], self.answers[pos:pos+10])
            count += 1





# def sigmoid(x, deriv=False):
#     if deriv:
#         return x * (1 - x)
#
#     return 1 / (1 + (np.exp(-x)))
#
# # inputs for testing
# X = np.array([[0., 0., 1.],
#               [0., 1., 1.]])
#
# # outputs for testing
# Y = np.array([[0., 1.],
#               [1., 0.]])
#
#
# weight1 = 2*np.random.random((3, 4))-1
# weight2 = 2*np.random.random((4, 2))-1
# weight3 = 2*np.random.random((2, 1))-1
#
# memes = sigmoid(np.dot(X, weight1))
# memes2 = sigmoid(np.dot(memes, weight2))
# memes3 = sigmoid(np.dot(memes2, weight3))
