import torch.nn as nn
import json
import torch

class LSTMEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=32):
        super(LSTMEmbedding, self).__init__()
        embedding_model = torch.load("/root/autodl-tmp/ETHGPT-main/large-scale-regression/tokengt/pretrain/nn_embedding_model.pth")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1).from_pretrained(embedding_model['weight'])
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
