# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

import pandas as pd
import numpy as np
import torch
import torch.nn as nn


torch.manual_seed(42)


class ECG_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super().__init__()

        proj_dim = input_dim

        #        self.in_proj = nn.Sequential(
        #            nn.Linear(input_dim, proj_dim),
        #            nn.ReLU(),
        #        )

        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=(0.0 if num_layers > 1 else 0.0),
        )

        # Output dimension adapts to uni- or bidirectional mode
        lstm_out_dim = hidden_dim
        # self.fc = nn.Linear(lstm_out_dim, 1)
        # simple nonlinear head

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim),
            nn.ReLU(),  # or nn.Tanh()
            nn.Linear(lstm_out_dim, 1),
        )

    def forward(self, x, hidden=None):
        # x: (B, T, F)
        #        x = self.in_proj(x) # (B, T, Proj)
        out, hidden = self.lstm(x, hidden)  # (B, T, H)
        logits = self.fc(out[:, -1, :])  # (B, 1)
        # logits = self.fc(out)  # (B, T, 1)
        return logits.squeeze(-1), hidden  # (B,)


# Early stopping class
class EarlyStopping:

    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
