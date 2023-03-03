import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional,
                 dropout_rate, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional,
                            dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids):
        # ids = [batch size, seq len]
        embedded = self.dropout(self.embedding(ids))
        # embedded = [batch size, seq len, embedding dim]
        packed_output, (hidden, cell) = self.lstm(embedded)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
            # hidden = [batch size, hidden dim * 2]
        else:
            hidden = self.dropout(hidden[-1])
            # hidden = [batch size, hidden dim]
        prediction = self.fc(hidden)
        # prediction = [batch size, output dim]
        return prediction


def evaluate_F1(pred, true):
    return metrics.f1_score(pred.cpu().detach().numpy(), true.cpu().detach().numpy(), average="weighted", zero_division=0)
def evaluate_accuracy(pred, true):
    return metrics.accuracy_score(pred.cpu().detach().numpy(), true.cpu().detach().numpy())
def evaluate_recall(pred, true):
    return metrics.recall_score(pred.cpu().detach().numpy(), true.cpu().detach().numpy(), average="weighted", zero_division=0)
def evaluate_precision(pred, true):
    return metrics.precision_score(pred.cpu().detach().numpy(), true.cpu().detach().numpy(), average="weighted", zero_division=0)

def train(epochs,dataloader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = epoch_accuracies = epoch_F1 = epoch_recall = epoch_precision = []
    for epoch in range(epochs):
        batch_losses = batch_accuracies = batch_F1 = batch_recall = batch_precision = []
        for review, label in dataloader:
            review = review.to(device)
            label = label.to(device)
            prediction = model(review)
            loss = criterion(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

            prediction = prediction.max(1)[1]
            batch_recall.append(evaluate_recall(prediction, label))
            batch_precision.append(evaluate_precision(prediction, label))
            batch_F1.append(evaluate_F1(prediction, label))
            batch_accuracies.append(evaluate_accuracy(prediction, label))

        epoch_losses.append(np.mean(batch_losses))
        epoch_recall.append(np.mean(batch_recall))
        epoch_precision.append(np.mean(batch_precision))
        epoch_F1.append(np.mean(batch_F1))
        epoch_accuracies.append(np.mean(batch_accuracies))

    return  epoch_losses, epoch_accuracies, epoch_F1, epoch_recall, epoch_precision


def test(dataloader, model, criterion, device):
    model.eval()
    losses = []
    accuracies = []
    F1 = []
    recall = []
    precision = []

    with torch.no_grad():
        for review, label in dataloader:
            review = review.to(device)
            label = label.to(device)
            prediction = model(review)
            loss = criterion(prediction, label)
            losses.append(loss.item())

            prediction = prediction.max(1)[1]
            recall.append(evaluate_recall(prediction, label))
            precision.append(evaluate_precision(prediction, label))
            F1.append(evaluate_F1(prediction, label))
            accuracies.append(evaluate_accuracy(prediction, label))

    return losses, accuracies, precision, F1, recall