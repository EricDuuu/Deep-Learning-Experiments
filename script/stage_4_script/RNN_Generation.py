import numpy as np
import torch

from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_CNN import Setting_CNN

if 1:
    np.random.seed(2)
    torch.manual_seed(2)

    dataset_dict = {1: 'MNIST', 2: 'ORL', 3: 'CIFAR'}
    # Change this when testing each individual datasets
    dataset = dataset_dict[3]

    data_obj = Dataset_Loader(dataset, '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = dataset

    method_obj = Method_CNN('convolutional neural network', '', dataset)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/CNN_'
    result_obj.result_destination_file_name = dataset

    setting_obj = Setting_CNN('CNN', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN Accuracy: ' + str(mean_score))
    print('************ Finish ************')
    # ------------------------------------------------------