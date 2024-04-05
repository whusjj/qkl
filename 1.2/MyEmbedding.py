import torch.nn as nn
import json
import torch

class LSTMEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=32):
        super(LSTMEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
    # def forward(self, x):
    #     embedded = self.embedding(x)
    #     output, (h_n, c_n) = self.lstm(embedded)
    #     return output

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return output[:, -1, :]
