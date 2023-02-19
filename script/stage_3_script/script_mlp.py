from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

# initialize objects
data_obj = Dataset_Loader('stage3', '')
data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
data_obj.dataset_source_file_name = 'ORL'


result_obj = Result_Saver('Result Saver', 'Saves prediction results')
result_obj.result_destination_folder_path = '/path/to/save/results/'
result_obj.result_destination_file_name = 'prediction_result'


evaluate_obj = Evaluate_Accuracy('Accuracy Evaluation', 'Evaluates accuracy of predictions')

method_obj = Method_CNN('','')
# prepare objects for running
method_obj.prepare(data_obj, None, evaluate_obj, result_obj)
method_obj.run()
# print overall performance
mean_score, std_score = evaluate_obj.get_mean_accuracy()
print('Overall Performance')
print('CNN Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
