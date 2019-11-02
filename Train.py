from Prepare_Data import *
from Interface import *
from Neural_Network import *
import numpy as np


def output(x):
    max = 0
    pos = None
    for i in range(x.shape[0]):
        if x[i] > max:
            max = x[i]
            pos = i
    answer = np.zeros(10)
    answer[pos] = 1
    return answer

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
    memes = train['train_images'][0]

    X = train['train_images'].reshape(train['train_images'].shape[0], 28*28) / 255.
    X_test = test['test_images'].reshape(test['test_images'].shape[0], 28*28) / 255.

    Y = train['train_labels']
    Y_test = test['test_labels']

    nn = NeuralNet(X, Y)
    nn.train()
    print("training complete:\n input test index: \n")
    # nn.save_training('Saved_Training', 'test')

    while True:
        pos = int(input())
        nn.forward_propagate(X_test[pos:pos+1])
        print(output(nn.output[0]))
        print(Y_test[pos])
        print()
