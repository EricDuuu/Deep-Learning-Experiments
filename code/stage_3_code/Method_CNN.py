'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import torch
from torch import nn

from code.base_class.method import method
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy


class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 1
    learning_rate = 1e-3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def __init__(self, mName, mDescription, dataset):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.dataset = dataset

        if self.dataset == 'MNIST':
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            ).to(self.device)
            self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, 5, 1, 2),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ).to(self.device)
            self.out = nn.Linear(32 * 7 * 7, 10).to(self.device)
        elif self.dataset == 'ORL':
            self.conv_layer_1 = nn.Conv2d(3, 32, 3, 1).to(self.device)
            self.activation_func_1 = nn.ReLU().to(self.device)
            self.conv_layer_2 = nn.Conv2d(32, 32, 3, 1).to(self.device)
            self.conv_layer_3 = nn.Conv2d(32, 64, 3, 1).to(self.device)
            self.fc_layer_1 = nn.Linear(3 * 2 * 64, 128).to(self.device)
            self.fc_layer_2 = nn.Linear(128, 40).to(self.device)
            self.activation_func_2 = nn.LogSoftmax(dim=1).to(self.device)
        elif self.dataset == 'CIFAR':
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=16,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            ).to(self.device)
            self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, 5, 1, 2),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ).to(self.device)
            self.out = nn.Linear(32 * 8 * 8, 10).to(self.device)

    def forward(self, x):
        if self.dataset == 'MNIST':
            x = self.conv1(x)
            x = self.conv2(x)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
            x = x.view(x.size(0), -1).to(self.device)
            output = self.out(x)
            return output
        elif self.dataset == 'ORL':
            h = self.activation_func_1(self.conv_layer_1(x))
            h = nn.MaxPool2d(5).to(self.device)(h)
            h = nn.ReLU().to(self.device)(self.conv_layer_2(h))
            h = nn.MaxPool2d(4).to(self.device)(h)
            h = nn.ReLU().to(self.device)(self.conv_layer_3(h))
            h = torch.flatten(h, 1)
            h = nn.ReLU().to(self.device)(self.fc_layer_1(h))
            y_pred = self.activation_func_2(self.fc_layer_2(h))
            return y_pred
        elif self.dataset == 'CIFAR':
            x = self.conv1(x)
            x = self.conv2(x)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
            x = x.view(x.size(0), -1).to(self.device)
            output = self.out(x)
            return output

    def train(self, data):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss().to(self.device)
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        for epoch in range(self.max_epoch):
            for i, (images, labels) in enumerate(data, 0):
                y_pred = self.forward(images.float().to(self.device))
                y_true = labels.to(self.device)

                optimizer.zero_grad()
                train_loss = loss_function(y_pred, y_true)
                train_loss.backward()
                optimizer.step()

            if epoch%1 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch,
                      'Accuracy:', accuracy_evaluator.evaluate_accuracy(),
                      'F1', accuracy_evaluator.evaluate_F1(),
                      'Precision', accuracy_evaluator.evaluate_precision(),
                      'Recall', accuracy_evaluator.evaluate_recall(),
                      'Loss:', train_loss.item()
                      )

    def test(self, data):
        y_pred_list = []
        y_true_list = []
        for i, (images, labels) in enumerate(data, 0):
            y_pred = self.forward(images.float())
            y_true = labels
            y_pred_list.append(y_pred.max(1)[1])
            y_true_list.append(y_true)
        y_true_list = torch.stack(y_true_list)
        y_pred_list = torch.stack(y_pred_list)
        y_true_list = torch.flatten(y_true_list)
        y_pred_list = torch.flatten(y_pred_list)

        return y_pred_list, y_true_list
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train'])
        print('--start testing...')
        pred_y, y_true = self.test(self.data['test'])
        return {'pred_y': pred_y, 'true_y': y_true}
            