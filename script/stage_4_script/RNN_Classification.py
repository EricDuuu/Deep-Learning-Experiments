import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torch.utils.data import TensorDataset, DataLoader

from code.stage_4_code.Data_Loader_Classification import Data_Loader
from code.stage_4_code.RNN_Model_Classification import LSTM, train, test

if 1:
    max_len = 256   # Max len of Sentences, to change, must delete old pickle to make a new instance

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
    layers = 2
    bidirectional = False
    dropout_rate = 0.5

    # Initialize Dataloaders for ease of batching
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize Model
    model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, layers, bidirectional, dropout_rate)

    # Embed the vectors
    vectors = torchtext.vocab.GloVe(name="6B", dim=embedding_dim)
    pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
    model.embedding.weight.data = pretrained_embedding

    # Model Hyperparameters
    epochs = 5
    lr = 0.001
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_function = loss_function.to(device)

    print("Starting Training")
    losses, accuracies, precisions, F1_scores, recalls = train(epochs, train_dataloader, model,
                                                               loss_function, optimizer, device)
    print("Training Ended")

    print("TRAINING")
    for i in range(epochs):
        print("epoch: ", i+1, "loss: ", losses[i], "accuracy: ", accuracies[i],
              "precision: ", precisions[i], "F1: ", F1_scores[i], "recall: ", recalls[i])

    print("Starting Testing")
    loss, accuracy, precision, F1_score,recall = test(test_dataloader, model, loss_function, device)
    print("Testing Ended")

    print("TESTING")
    loss, accuracy, precision, F1_score,recall = np.mean(loss), np.mean(accuracy), np.mean(precision), \
        np.mean(F1_score),np.mean(recall)
    print("loss: ", loss, "accuracy: ", accuracy, "precision: ", precision,
          "F1: ", F1_score, "recall: ", recall)