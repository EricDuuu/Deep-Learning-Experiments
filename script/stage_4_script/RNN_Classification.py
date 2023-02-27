import numpy as np
import torch

from code.stage_4_code.Dataset_Loader_Classification import Dataset_Loader
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_4_code.Method_RNN_Classification import Method_RNN
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting_RNN import Setting_RNN

if 1:
    np.random.seed(2)
    torch.manual_seed(2)

    data_obj = Dataset_Loader('Classification', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/'
    data_obj.dataset_source_file_name = 'text_classification'

    method_obj = Method_RNN('convolutional neural network', '', 'Classification')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/RNN_'
    result_obj.result_destination_file_name = 'Classification'

    setting_obj = Setting_RNN('RNN', '')

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