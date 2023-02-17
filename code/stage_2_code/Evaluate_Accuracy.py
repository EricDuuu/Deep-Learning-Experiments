'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from sklearn import metrics

from code.base_class.evaluate import evaluate


class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate_F1(self):
        return metrics.f1_score(self.data['true_y'], self.data['pred_y'], average="weighted", zero_division=0)
    def evaluate_accuracy(self):
        return metrics.accuracy_score(self.data['true_y'], self.data['pred_y'])

    def evaluate_recall(self):
        return metrics.recall_score(self.data['true_y'], self.data['pred_y'], average="weighted", zero_division=0)
    def evaluate_precision(self):
        return metrics.precision_score(self.data['true_y'], self.data['pred_y'], average="weighted", zero_division=0)