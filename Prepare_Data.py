import os, codecs, numpy, gzip, shutil
from skimage.io import imsave
import urllib.request


class PrepareData:

    def __init__(self, path='Data/MINSTData/'):
        self.data_path = path
        self.download_data()
        self.dataset = self.read_files()

    # Convert 4 bytes of data into an integer
    @staticmethod
    def __bytes_to_int(bytes):
        return int(codecs.encode(bytes, 'hex'), 16)

    # Download the dataset from the MINST dataset
    def download_data(self):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',  # URLS to download the MINST dataset from
                'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']

        for url in urls:
            filename = url.split('/')[-1]  # Split the URL at the '/', and then store the last string in the filename variable

            if os.path.exists(self.data_path + filename) or os.path.exists(self.data_path + filename.split('.')[0]):
                print(filename, 'already exists')
            else:
                print('Downloading ', filename)
                urllib.request.urlretrieve(url, self.data_path + filename)

        print('\n************************************')
        print('All files have have been downloaded')
        print('************************************\n')

        files = os.listdir(self.data_path)
        for file in files:
            if os.path.exists(self.data_path + file.split('.')[0]):
                print(file, 'already exists')
            else:
                if file.endswith('gz'):
                    print('Extracting: ', file)
                    with gzip.open(self.data_path + file, 'rb') as f_in:
                        with open(self.data_path + file.split('.')[0], 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

        print('\n************************************')
        print('All files have have been extracted')
        print('************************************\n')

        for file in files:
            if file.endswith('gz'):
                print('Removing ', file)
                os.remove(self.data_path + file)
        print('All Archives have been removed')

    # Store data within a dictionary
    def read_files(self):
        dataset = {}
        files = os.listdir(self.data_path)  # Directory for the download of the MINST dataset
        for file in files:
            if file.endswith('ubyte'):  # If file in the directory belongs to the dataset
                print('\nReading file: ', file)

                with open(self.data_path+file, 'rb') as f_open:  # Open file in binary read mode
                    data = f_open.read()
                    file_type = self.__bytes_to_int(data[:4])  # File type specifies if the file is for images or lables
                    data_length = self.__bytes_to_int(data[4:8])  # Total number of data entries

                    if file_type == 2051:  # File contains image data
                        category = 'images'
                        num_rows = self.__bytes_to_int(data[8:12])  # Number of rows of pixels in the image
                        num_cols = self.__bytes_to_int(data[12:16])  # Number of cols of pixels in the image

                        parsed_data = numpy.frombuffer(data, dtype=numpy.uint8, offset=16)
                        parsed_data = parsed_data.reshape(data_length, num_rows, num_cols)

                    elif(file_type == 2049):  # File contains lable data
                        category = 'labels'
                        parsed_data = numpy.frombuffer(data, dtype=numpy.uint8, offset=8)
                        parsed_data = parsed_data.reshape(data_length)

                    if(data_length == 10000):
                        set = 'test'

                    elif(data_length == 60000):
                        set = 'train'

                    dataset[set+'_'+category] = parsed_data
        return dataset

    # Parse the data within @dataset, storing it within the given directory in labeled files dependant on class
    def store_data(self):
        sets = ['train', 'test']
        for set in sets:
            images = self.dataset[set+'_images']
            labels = self.dataset[set+'_labels']
            num_samples = images.shape[0]
            for index in range(num_samples):
                print(set,': ', index)
                image = images[index]
                label = labels[index]
                if not os.path.exists(self.data_path+set+'/'+str(label)+'/'):
                    os.makedirs(self.data_path+set+'/'+str(label)+'/')
                filenumber = len(os.listdir(self.data_path+set+'/'+str(label)+'/'))
                imsave(self.data_path+set+'/'+str(label)+'/%05d.png'%(filenumber), image)


prepare = PrepareData()
#prepare.store_data()