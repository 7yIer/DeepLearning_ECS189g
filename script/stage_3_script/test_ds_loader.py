from code.stage_3_code.Dataset_Loader import Dataset_Loader
#from code.stage_3_code.Method_MLP import Method_MLP
from code.stage_3_code.Result_Saver import Result_Saver
#from code.stage_3_code.Setting_KFold_CV import Setting_KFold_CV
#from code.stage_3_code.Setting_Train_Test import Setting_Train_Test
#from code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
from code.base_class.dataset import dataset


import pickle
import matplotlib.pyplot as plt


# def main():
#     f = open('../../data/stage_3_data/', 'rb')  # or change MNIST to other dataset names
#     data = pickle.load(f)
#     f.close()
#     print('training set size:', len(data['train']), 'testing set size:',
#           len(data['test']))
#     for pair in data['train']:
#     # for pair in data['test']:
#         plt.imshow(pair['image'], cmap="Greys")
#         plt.show()
#         print(pair['label'])
class MyDataset(dataset):
    def __init__(self, data):
        self.images = []
        self.labels = []
        for item in data:
            self.images.append(item['image'])
            self.labels.append(item['label'])

    def __repr__(self):
        return str(tuple((self.images, self.labels)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def main():
    if 1:

        dataset_type = "CIFAR"
        with open('../../data/stage_3_data/' + dataset_type, 'rb') as f:
            data = pickle.load(f)

        train_dataset_orl = MyDataset(data['train'])
        test_dataset_orl = MyDataset(data['test'])

        print("This is training data for orl")
        print(train_dataset_orl)
        print("*\n" * 40)

        print("This is testing data for orl")
        print(test_dataset_orl)


        # # ---- objection initialization setction ---------------
        # data_obj = Dataset_Loader('MNIST', '')
        # data_obj.dataset_source_file_name = 'MNIST'
        # data_obj.dataset_source_folder_path = '../../data/stage_3_data/'

        #dict1 = data_obj.load()
        #print(dict1)

        # data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
        # data_obj.dataset_source_file_name = 'MNIST'
        #
        # method_obj = Method_SVM('support vector machine', '')
        # method_obj.c = c
        #
        # result_obj = Result_Saver('saver', '')
        # result_obj.result_destination_folder_path = '../../result/stage_3_result/SVM_'
        # result_obj.result_destination_file_name = 'prediction_result'
        #
        # # setting_obj = Setting_KFold_CV('k fold cross validation', '')
        # # setting_obj = Setting_Train_Test_Split('train test split', '')
        #
        # evaluate_obj = Evaluate_Accuracy('accuracy', '')
        # # ------------------------------------------------------
        #
        # # ---- running section ---------------------------------
        # print('************ Start ************')
        # setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        # setting_obj.print_setup_summary()
        # mean_score, std_score = setting_obj.load_run_save_evaluate()
        # print('************ Overall Performance ************')
        # print('SVM Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
        # print('************ Finish ************')
        # # ------------------------------------------------------


if __name__ == '__main__':
    main()