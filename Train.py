from Prepare_Data import *
from Interface import *
from Neural_Network import *


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

    nn = NeuralNet(28*28, 32, 10)
    nn.train(X, Y)
    # nn.save_training('Saved_Training/', 'test')
    # nn.load_training('Saved_Training/', 'training')

    while True:
        pos = int(input())
        nn.forward_propagate(X_test[pos:pos+1])
        print(nn.net_output(nn.output[0]))
        print(Y_test[pos])
        print()
