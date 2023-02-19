import numpy as np
import json
from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r') as f:
            data_dict = json.load(f)

        train_data = data_dict['train']
        test_data = data_dict['test']

        train_X = [np.array(data['image']) for data in train_data]
        train_y = [data['label'] for data in train_data]

        test_X = [np.array(data['image']) for data in test_data]
        test_y = [data['label'] for data in test_data]

        return {'train_X': train_X, 'train_y': train_y, 'test_X': test_X, 'test_y': test_y}
