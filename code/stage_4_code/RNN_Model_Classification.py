import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, layers, bidirectional, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers, bidirectional=bidirectional,
                            dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, reviews):
        embedded = self.dropout(self.embedding(reviews))
        output, (hidden, cell) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        else:
            hidden = self.dropout(hidden[-1])
        prediction = self.fc(hidden)
        return prediction


def evaluate_F1(pred, true):
    return metrics.f1_score(pred.cpu().detach().numpy(), true.cpu().detach().numpy(), average="weighted", zero_division=0)
def evaluate_accuracy(pred, true):
    return metrics.accuracy_score(pred.cpu().detach().numpy(), true.cpu().detach().numpy())
def evaluate_recall(pred, true):
    return metrics.recall_score(pred.cpu().detach().numpy(), true.cpu().detach().numpy(), average="weighted", zero_division=0)
def evaluate_precision(pred, true):
    return metrics.precision_score(pred.cpu().detach().numpy(), true.cpu().detach().numpy(), average="weighted", zero_division=0)

def train(epochs, dataloader, model, loss_function, optimizer, device):
    BCE = loss_function._get_name() == "BCEWithLogitsLoss" or loss_function._get_name() == "BCELoss"
    model.train()
    losses = []
    accuracies = []
    F1s = []
    recalls = []
    precisions = []
    for epoch in range(epochs):
        for review, label in dataloader:
            review = review.to(device)
            if BCE:
                label = label.float()
            label = label.to(device)
            prediction = model(review)
            if BCE:
                prediction = torch.round(nn.Sigmoid()(prediction).squeeze())

            loss = loss_function(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if not BCE:
                prediction = prediction.max(1)[1]
            recalls.append(evaluate_recall(prediction, label))
            precisions.append(evaluate_precision(prediction, label))
            F1s.append(evaluate_F1(prediction, label))
            accuracies.append(evaluate_accuracy(prediction, label))

        loss = np.mean(losses)
        recall = np.mean(recalls)
        precision = np.mean(precisions)
        F1 = np.mean(F1s)
        accuracy = np.mean(accuracies)
        print("epoch: ", epoch+1, "loss: ", loss, "accuracy: ", accuracy, "precision: ",
              precision, "F1: ", F1, "recall: ", recall)

    return (losses, accuracies, F1s, recalls, precisions)


def test(dataloader, model, loss_function, device):
    BCE = loss_function._get_name() == "BCEWithLogitsLoss" or loss_function._get_name() == "BCELoss"
    model.eval()
    losses = []
    accuracies = []
    F1 = []
    recall = []
    precision = []

    with torch.no_grad():
        for review, label in dataloader:
            review = review.to(device)

            if BCE:
                label = label.float()

            label = label.to(device)
            prediction = model(review)

            if BCE:
                prediction = torch.round(nn.Sigmoid()(prediction).squeeze())

            loss = loss_function(prediction, label)
            losses.append(loss.item())

            if not BCE:
                prediction = prediction.max(1)[1]
            recall.append(evaluate_recall(prediction, label))
            precision.append(evaluate_precision(prediction, label))
            F1.append(evaluate_F1(prediction, label))
            accuracies.append(evaluate_accuracy(prediction, label))

    return (losses, accuracies, precision, F1, recall)