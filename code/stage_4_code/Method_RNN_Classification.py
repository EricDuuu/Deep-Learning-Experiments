'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import numpy as np
import torch
from torch import nn

from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy


class Method_RNN(method, nn.Module):
    data = None
    max_epoch = 500
    batch_size = 4096
    learning_rate = 1e-3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.fc_layer_1 = nn.Linear(784, 256)
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(256, 128)
        self.activation_func_2 = nn.ReLU()

        # self.activation_func_2 = nn.Softmax(dim=1)

        self.fc_layer_3 = nn.Linear(128, 10)
        self.activation_func_3 = nn.Softmax(dim=1)

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h = self.activation_func_1(self.fc_layer_1(x))
        h2 = self.activation_func_2(self.fc_layer_2(h))
        y_pred = self.activation_func_3(self.fc_layer_3(h2))
        return y_pred

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        for epoch in range(self.max_epoch):
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            y_true = torch.LongTensor(np.array(y))
            train_loss = loss_function(y_pred, y_true)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch,
                      'Accuracy:', accuracy_evaluator.evaluate_accuracy(),
                      'F1', accuracy_evaluator.evaluate_F1(),
                      'Precision', accuracy_evaluator.evaluate_precision(),
                      'Recall', accuracy_evaluator.evaluate_recall(),
                      'Loss:', train_loss.item()
                      )

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
