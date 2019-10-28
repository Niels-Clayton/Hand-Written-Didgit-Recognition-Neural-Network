import Prepare_Data as load

if __name__ == '__main__':
    load_data = load.PrepareData()
    load_data.read_files()

    # retrieve the training dataset from the downloaded binary files
    train = load_data.get_training
