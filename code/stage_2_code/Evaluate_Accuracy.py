'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from sklearn import metrics

from code.base_class.evaluate import evaluate


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        return metrics.mean_squared_error(self.data['true_y'], self.data['pred_y'])
