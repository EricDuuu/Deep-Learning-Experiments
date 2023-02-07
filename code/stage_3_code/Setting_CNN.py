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

        # X_train, X_test, y_train, y_test = train_test_split(loaded_data['X'], loaded_data['y'], test_size = 0.33)

        # run MethodModule
        self.method.data = loaded_data
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate()

        