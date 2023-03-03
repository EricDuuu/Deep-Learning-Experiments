import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torch.utils.data import TensorDataset, DataLoader

from code.stage_4_code.Data_Loader_Classification import Data_Loader
from code.stage_4_code.RNN_Model_Classification import LSTM, train, test

if 1:
    max_len = 256   # Max len of Reviews
    # Returns encoded and padded dataset
    train_data, test_data, vocab = Data_Loader(max_len).run()
    train_data = TensorDataset(torch.LongTensor(train_data['X']), torch.LongTensor(train_data['Y']))
    test_data = TensorDataset(torch.LongTensor(test_data['X']), torch.LongTensor(test_data['Y']))

    # Model Parameters
    batch_size = 512
    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_dim = 300
    output_dim = 2  # 0 and 1
    n_layers = 2
    bidirectional = False
    dropout_rate = 0.5

    # Initialize Dataloaders for ease of batching
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Initialize Model
    model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate, 1)


    vectors = torchtext.vocab.GloVe(name="6B", dim=100)
    pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
    model.embedding.weight.data = pretrained_embedding

    # Model Hyperparameters
    lr = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    epochs = 10

    losses, accuracies, precision, F1, recall = train(epochs,train_dataloader, model, criterion, optimizer, device)
    for i in range(epochs):
        print("epoch: ", i+1, "loss: ", losses[i], "accuracy: ", accuracies[i], "precision: ", precision[i], "F1: ", F1[i], "recall: ", recall[i])
    nice = test(test_dataloader, model, criterion, device)
