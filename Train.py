from Prepare_Data import *
from Interface import *
from Neural_Network import *
import numpy as np


if __name__ == '__main__':

    load_data = Prepare_Data()

    app = wx.App()
    frame = DownloadButton()
    app.MainLoop()
    if frame.download:
        load_data.download_data()

    load_data.read_files()
    # retrieve the training dataset from the downloaded binary files
    test = load_data.get_test
    train = load_data.get_training

    X = train['train_images'].reshape(train['train_images'].shape[0], 28*28)
    X = X / 255.
    X_test = test['test_images'].reshape(test['test_images'].shape[0], 28 * 28)
    X_test = X_test / 255.

    Y = train['train_labels']
    Y_test = test['test_labels']

    nn = NeuralNet(X, Y)
    nn.train()

    # nn = NeuralNet(X[0:50], Y[0:50])
    # nn.back_propagate(X[0:50], Y[0:50])
    # while np.mean(np.abs(nn.output_error)) > 1:
    #     print(np.mean(np.abs(nn.output_error)))
    #     nn.back_propagate(X[0:50], Y[0:50])
    #
    # print(nn.output)

