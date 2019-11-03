import numpy as np
import scipy.special as sci
import os


class NeuralNet:
    def __init__(self, training_set, answer_set):

        self.inputs = training_set
        self.answers = answer_set
        self.lr = 0.1

        self.input_size = training_set.shape[1]
        self.hidden_layer_size = 38
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
            # return sci.expit(x) * (1 - sci.expit(x))
            return x * (1-x)

        return sci.expit(x)

    @staticmethod
    def net_output(x):
        max_val = 0
        pos = None
        for i in range(x.shape[0]):
            if x[i] > max_val:
                max_val = x[i]
                pos = i
        answer = np.zeros(10, dtype=np.float)
        answer[pos] = 1
        return answer

    def forward_propagate(self, X):
        self.layer_1_output = self.sigmoid(np.dot(X, self.hidden_weights_1))
        self.output = self.sigmoid(np.dot(self.layer_1_output, self.output_weights))

    def calculate_error(self):
        self.output_error = (self.correct_output - self.output)
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

        while self.total_error > 1e-4:
            if count % 1000 == 0:
                print(np.mean(np.abs(self.output_error)))

            if count > 10000000:
                break

            pos = np.random.randint(self.inputs.shape[0]-10)
            self.back_propagate(self.inputs[pos:pos+10], self.answers[pos:pos+10])
            count += 1

    def save_training(self, dir, file_name):
        if not os.path.exists(dir):
            os.makedirs(dir)

        if os.path.exists(dir + file_name + '.npz'):
            print(file_name + " already exists\n")
            print("Please enter a new file name:    ")
            file_name = input()

        save_path = dir + '/' + file_name
        np.savez(save_path, hidden=self.hidden_weights_1, output=self.output_weights)

    def load_training(self, dir, file_name):
        file_path = dir + file_name + '.npz'
        if not os.path.exists(file_path):
            print("This file does not exist.")
            return
        saved = np.load(file_path)
        self.hidden_weights_1 = saved['hidden']
        self.output_weights = saved['output']
        saved.close()
        return
