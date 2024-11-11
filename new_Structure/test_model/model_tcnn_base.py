import torch
import torch.nn as nn
from math import ceil


class nnBatch_Norm_2D(nn.Module):
    def __init__(
            self,
            dim1 : int,
            dim2 : int
    ):
        super(nnBatch_Norm_2D, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.batch_norm = nn.BatchNorm1d(self.dim1 * self.dim2)

    def forward(self, x):
        return self.batch_norm(x.contiguous().view(-1, self.dim1 * self.dim2)).view(-1, self.dim1, self.dim2)

class nnPermute(nn.Module):
    def __init__(self, dim0, dim1, dim2):
        super().__init__()
        self.dim0, self.dim1, self.dim2 = dim0, dim1, dim2

    def forward(self, x):
        return x.permute(self.dim0, self.dim1, self.dim2)

class TCNN_base(nn.Module):
    def __init__(self, WINDOW_SIZE, FC_DROPOUT):
        super(TCNN_base, self).__init__()
        self.name = 'TCNN'

        self.TCNN = nn.Sequential(
            nn.Conv1d(
                in_channels=WINDOW_SIZE,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            # nn.Dropout(p=TCNN_DROPOUT),
            nnBatch_Norm_2D(32, 14),

            nnPermute(0, 2, 1),
            nn.MaxPool1d(
                kernel_size=17,
                stride=1
            ),
            nnPermute(0, 2, 1),

            nn.Conv1d(
                in_channels=16,
                out_channels=8,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            # nn.Dropout(p=TCNN_DROPOUT),
            nnBatch_Norm_2D(8, 14),

            nnPermute(0, 2, 1),
            nn.AvgPool1d(
                kernel_size=5,
                stride=1
            ),
            nnPermute(0, 2, 1),

            nn.Conv1d(
                in_channels=4,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            # nn.Dropout(p=TCNN_DROPOUT)
            nnBatch_Norm_2D(1, 14)
        )

        self.FC = nn.Sequential(
            nn.Linear(
                in_features=14,
                out_features=16
            ),
            nn.ReLU(),
            nn.Dropout(p=FC_DROPOUT),

            nn.Linear(
                in_features=16,
                out_features=8
            ),
            nn.ReLU(),
            nn.Dropout(p=FC_DROPOUT),

            nn.Linear(
                in_features=8,
                out_features=1
            )
        )


    def forward(self, X): # X(batch_size, window_size, eff_sensors)
        X_TCNN = self.TCNN(X)

        X_TCNN = X_TCNN.view(-1, 14)
        out = self.FC(X_TCNN)

        return out
