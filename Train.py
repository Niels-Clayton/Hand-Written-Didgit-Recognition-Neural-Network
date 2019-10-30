from Prepare_Data import *
from Interface import *

if __name__ == '__main__':

    load_data = Prepare_Data()

    app = wx.App()
    frame = DownloadButton()
    app.MainLoop()
    if frame.download:
        load_data.download_data()


    load_data.read_files()
    # retrieve the training dataset from the downloaded binary files
    train = load_data.get_training
