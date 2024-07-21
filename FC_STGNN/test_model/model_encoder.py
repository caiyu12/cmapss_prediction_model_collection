import torch
import torch.nn as nn
from math import ceil

SCALING_VALUE = 1

BATCH_SIZE  = 32
EFF_SENSORS = 14
# EFF_SENSORS=14 + 3
FEATURE_NUM = EFF_SENSORS + ceil(EFF_SENSORS/16)

WINDOW_SIZE = 50
SLIDING_STEP = 1 # positive integer


N_HEAD = 4
ENCODER_LAYER = 3
ENCODER_FFN_UNIT = 64


LEARNING_RATE = 1E-4
DROPOUT = 0.5
CNN_LAYER_NUM = 4


N_EPOCHS = 500

class Batch_Norm_2D(nn.Module):
    def __init__(
            self,
            dim1 : int,
            dim2 : int
    ):
        super(Batch_Norm_2D, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.batch_norm = nn.BatchNorm1d(self.dim1 * self.dim2)

    def forward(self, x):
        return self.batch_norm(x.contiguous().view(-1, self.dim1 * self.dim2)).view(-1, self.dim1, self.dim2)

class Permute(nn.Module):
    def __init__(self, dim0, dim1, dim2):
        super().__init__()
        self.dim0, self.dim1, self.dim2 = dim0, dim1, dim2

    def forward(self, x):
        return x.permute(self.dim0, self.dim1, self.dim2)

class TCNN_TransEncoder(nn.Module):
    def __init__(self):
        super(TCNN_TransEncoder, self).__init__()

        __encoder_layer = nn.TransformerEncoderLayer(
            batch_first=True,
            nhead=N_HEAD,

            d_model=EFF_SENSORS*WINDOW_SIZE,
            dim_feedforward=ENCODER_FFN_UNIT,

            dropout=DROPOUT,
            activation='relu',

            # bias=True
        )
        self.Trans_Encoder = nn.TransformerEncoder(
            __encoder_layer,
            num_layers=ENCODER_LAYER
        )

        self.TCNN = nn.Sequential(
            #input shape: (BATCH_SIZE, WINDOW_SIZE, EFF_SENSORS)

            #layer0
            nn.Conv1d(
                in_channels=WINDOW_SIZE,
                out_channels=WINDOW_SIZE,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            Batch_Norm_2D(WINDOW_SIZE, EFF_SENSORS),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            #output shape: (BATCH_SIZE, WINDOW_SIZE, 14) if EFF_SENSORS=14

            #layer1
            nn.Conv1d(
                in_channels=WINDOW_SIZE,
                out_channels=WINDOW_SIZE*2,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            Batch_Norm_2D(WINDOW_SIZE*2, ceil(EFF_SENSORS/2)),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            #output shape: (BATCH_SIZE, WINDOW_SIZE*2, 7) if EFF_SENSORS=14

            #layer2
            nn.Conv1d(
                in_channels=WINDOW_SIZE*2,
                out_channels=WINDOW_SIZE*2*2,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            Batch_Norm_2D(WINDOW_SIZE*2*2, ceil(EFF_SENSORS/2/2)),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            #output shape: (BATCH_SIZE, WINDOW_SIZE*2*2, 4) if EFF_SENSORS=14

            #layer3
            nn.Conv1d(
                in_channels=WINDOW_SIZE*2*2,
                out_channels=WINDOW_SIZE*2*2*2,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            Batch_Norm_2D(WINDOW_SIZE*8, ceil(EFF_SENSORS/8)),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            #output shape: (BATCH_SIZE, WINDOW_SIZE*2*2*2, 2) if EFF_SENSORS=14

            #layer4
            nn.Conv1d(
                in_channels=WINDOW_SIZE*8,
                out_channels=WINDOW_SIZE*16,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            Batch_Norm_2D(WINDOW_SIZE*16, ceil(EFF_SENSORS/16)),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            #output shape: (BATCH_SIZE, WINDOW_SIZE*2*2*2*2, 1) if EFF_SENSORS=14

            Permute(0, 2, 1), # out = (BATCH_SIZE, 1, WINDOW_SIZE*2*2*2*2) if EFF_SENSORS=14
            #AvgPool-1D
            nn.AvgPool1d(
                kernel_size=16 # 16 = 2*2*2*2
            ),
            Permute(0, 2, 1) # out = (BATCH_SIZE, WINDOW_SIZE, 1) if EFF_SENSORS=14
        )

        # feature recalibration mechanism
        self.FRM = nn.Sequential(
            nn.Linear(
                in_features=FEATURE_NUM,
                out_features=1,
            ),
            nn.ReLU(),

            nn.Linear(
                in_features=1,
                out_features=FEATURE_NUM
            ),
            nn.Sigmoid()
        )

        self.out = nn.Sequential(
            nn.Linear(
                in_features=WINDOW_SIZE*(EFF_SENSORS+1),
                out_features=64
            ),
            nn.ReLU(),

            nn.Linear(
                in_features=64,
                out_features=1
            ),
            nn.ReLU()
        )

    def forward(self, X):
        X_encoder_in  = X.contiguous().view(-1, WINDOW_SIZE*EFF_SENSORS)
        X_encoder_out = self.Trans_Encoder(X_encoder_in)
        X_encoder_out = X_encoder_out.contiguous().view(-1, WINDOW_SIZE, EFF_SENSORS) # out = (BATCH_SIZE, WINDOW_SIZE*EFF_SENSORS)

        X_tcnn_out    = self.TCNN(X) # out = (BATCH_SIZE, WINDOW_SIZE, 1) if EFF_SENSORS=14

        X_encoder_out = X_encoder_out.permute(0, 2, 1) # out = (BATCH_SIZE, EFF_SENSORS, WINDOW_SIZE)
        X_tcnn_out    = X_tcnn_out.permute(0, 2, 1)    # out = (BATCH_SIZE, 1, WINDOW_SIZE)
        X_concat = torch.cat((X_encoder_out, X_tcnn_out), dim=1) # out = (BATCH_SIZE, EFF_SENSORS+1, WINDOW_SIZE)

        X_concat = X_concat.permute(0, 2, 1) # out = (BATCH_SIZE, WINDOW_SIZE, EFF_SENSORS+1)
        X_frm_out = self.FRM(X_concat) # out = (BATCH_SIZE, WINDOW_SIZE, EFF_SENSORS+1)

        X_out = torch.mul(X_concat, X_frm_out)
        X_out = X_out.contiguous().view(-1, WINDOW_SIZE*(EFF_SENSORS+1))

        out = self.out(X_out)

        return out
