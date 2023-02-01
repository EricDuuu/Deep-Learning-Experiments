'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    test_data = None
    test_path = None
    test_file_name = None

    train_data = None
    train_path = None
    train_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        X_test = []
        y_test = []

        X_train = []
        y_train = []

        f = open(self.test_path + self.test_file_name, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            X_test.append(elements[1:])
            y_test.append(elements[0])
        f.close()

        f = open(self.train_path + self.train_file_name, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            X_train.append(elements[1:])
            y_train.append(elements[0])
        f.close()

        return {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
