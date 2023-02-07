'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import numpy as np
import torch
from torch import nn

from code.base_class.method import method
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy


class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 500
    learning_rate = 1e-3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def __init__(self, mName, mDescription, dataset):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        if dataset == 'MNIST':
            self.fc_layer_1 = nn.Linear(4, 4).to(self.device)
            self.activation_func_1 = nn.ReLU()
            self.fc_layer_2 = nn.Linear(4, 2).to(self.device)
            self.activation_func_2 = nn.Softmax(dim=1)
        elif dataset == 'ORL':
            self.fc_layer_1 = nn.Linear(4, 4).to(self.device)
            self.activation_func_1 = nn.ReLU()
            self.fc_layer_2 = nn.Linear(4, 2).to(self.device)
            self.activation_func_2 = nn.Softmax(dim=1)
        elif dataset == 'CIFAR':
            self.fc_layer_1 = nn.Linear(4, 4).to(self.device)
            self.activation_func_1 = nn.ReLU()
            self.fc_layer_2 = nn.Linear(4, 2).to(self.device)
            self.activation_func_2 = nn.Softmax(dim=1)


    def forward(self, x):
        h = self.activation_func_1(self.fc_layer_1(x))
        y_pred = self.activation_func_2(self.fc_layer_2(h))
        return y_pred
    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        for epoch in range(self.max_epoch):
            y_pred = self.forward(torch.FloatTensor(np.array(X)).to(self.device))
            y_true = torch.LongTensor(np.array(y)).to(self.device)
            train_loss = loss_function(y_pred, y_true)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if epoch%100 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
    
    def test(self, X):
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            