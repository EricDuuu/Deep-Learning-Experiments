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
    max_epoch = 10
    learning_rate = 1e-3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, mName, mDescription, dataset):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.dataset = dataset

        if self.dataset == 'MNIST':
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2, ),
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
            self.Normal_ORL(3, 40, 28, 23)

        elif self.dataset == 'CIFAR':
            self.Normal_CIFAR()

    def Normal_CIFAR(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        ).to(self.device)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ).to(self.device)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ).to(self.device)
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        ).to(self.device)
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        ).to(self.device)
        self.fc3 = nn.Linear(512, 10).to(self.device)

    def Normal_CIFARf(self, x):
        x = self.conv1(x).to(self.device)
        x = self.conv2(x).to(self.device)
        x = self.conv3(x).to(self.device)
        x = x.view(x.size(0), -1).to(self.device)
        x = self.fc1(x).to(self.device)
        x = self.fc2(x).to(self.device)
        output = self.fc3(x).to(self.device)
        return output

    def Normal_ORL(self, inchannels=3, classes=10, x=8, y=8):
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=16, kernel_size=5, stride=1, padding=2, ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        ).to(self.device)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ).to(self.device)

        self.out = nn.Linear(32 * x * y, classes).to(self.device)

    def Normal_ORLF(self, x):
        x = self.conv1(x).to(self.device)
        x = self.conv2(x).to(self.device)
        x = x.view(x.size(0), -1).to(self.device)
        output = self.out(x).to(self.device)
        return output

    def forward(self, x):
        if self.dataset == 'MNIST':
            x = self.conv1(x).to(self.device)
            x = self.conv2(x).to(self.device)
            x = x.view(x.size(0), -1).to(self.device)
            output = self.out(x).to(self.device)
            return output
        elif self.dataset == 'ORL':
            return self.Normal_ORLF(x)
        elif self.dataset == 'CIFAR':
            return self.Normal_CIFARf(x)

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

            if epoch % 1 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch + 1,
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
            