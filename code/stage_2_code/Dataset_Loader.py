'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        X = []
        y = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        c = 0
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            X.append(elements[1:])
            y.append(elements[0])

        f.close()

        XTest = []
        yTest = []

        f = open(self.dataset_source_folder_path + "test.csv", 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            XTest.append(elements[1:])
            yTest.append(elements[0])
        f.close()


        return {'X': X, 'y': y}, {'X': XTest, 'y': yTest}