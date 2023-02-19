'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting


class Setting_CNN(setting):

    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()

        # run MethodModule
        self.method.data = loaded_data
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result

        print("TESTING RESULTS")
        print(
            'Accuracy:', self.evaluate.evaluate_accuracy(),
            'F1', self.evaluate.evaluate_F1(),
            'Precision', self.evaluate.evaluate_precision(),
            'Recall', self.evaluate.evaluate_recall(),
        )
        return self.evaluate.evaluate_accuracy()

        