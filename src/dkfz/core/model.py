# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

import copy
import torch
import torch.nn as nn

torch.manual_seed(42)


class ECG_RNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        num_layers=1,
        recovery_head=False,
        prediction_delay_frames=0,
    ):
        super().__init__()

        if prediction_delay_frames not in (0, 1) or isinstance(
            prediction_delay_frames, bool
        ):
            raise ValueError("prediction_delay_frames must be 0 or 1")
        if prediction_delay_frames and recovery_head:
            raise ValueError("A delayed primary cannot be combined with recovery_head")
        self.prediction_delay_frames = int(prediction_delay_frames)
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
        self.recovery_head = bool(recovery_head)
        if self.recovery_head:
            self.recovery_fc = copy.deepcopy(self.fc)

    def load_checkpoint(self, state_dict, *, initialize_recovery=False):
        """Load strictly, or explicitly add a recovery head to a legacy model."""
        has_recovery = any(key.startswith("recovery_fc.") for key in state_dict)
        if initialize_recovery and self.recovery_head and not has_recovery:
            state_dict = dict(state_dict)
            for key, value in list(state_dict.items()):
                if key.startswith("fc."):
                    state_dict["recovery_" + key] = value.clone()
        self.load_state_dict(state_dict)

    def freeze_primary(self):
        """Train the recovery head while preserving all original detector weights."""
        if not self.recovery_head:
            raise ValueError("freeze_primary requires recovery_head=True")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("recovery_fc."))

    def forward(self, x, hidden=None):
        # x: (B, T, F)
        #        x = self.in_proj(x) # (B, T, Proj)
        out, hidden = self.lstm(x, hidden)  # (B, T, H)
        # Delayed targets still use the final state, including the context frame.
        representation = out[:, -1, :]
        logits = self.fc(representation)  # (B, 1)
        if self.recovery_head:
            # Both heads see the complete window through t. Output 1 predicts
            # t-1, so it has one frame of additional evidence after its target.
            return torch.cat((logits, self.recovery_fc(representation)), dim=-1), hidden
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
