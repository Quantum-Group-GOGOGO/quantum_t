import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
# ----------------------
# 定义 Aux Autoencoder
# ----------------------
class Auxencoder(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=64, proj_dim=100):
        super().__init__()
        # Encoder: 9 -> hidden_dim -> proj_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        z = self.encoder(x)
        return z

class Auxdecoder(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=64, proj_dim=100):
        super().__init__()
        # Decoder: proj_dim -> hidden_dim -> in_dim
        self.decoder = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim)
        )
    def forward(self, z):
        x_rec = self.decoder(z)
        return x_rec


class AuxAutoencoder(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=64, proj_dim=100):
        super().__init__()
        # Encoder: 9 -> hidden_dim -> proj_dim
        self.encoder = Auxencoder(in_dim, hidden_dim, proj_dim)
        # Decoder: proj_dim -> hidden_dim -> in_dim
        self.decoder = Auxdecoder(in_dim, hidden_dim, proj_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec
