import torch
import torch.nn as nn

EFF_SENSORS = 14
WINDOW_SIZE = 30
MLP_FEATURE_OUTSIZE = 30
DROPOUT = 0.2
MIXING_LAYERS = 8
# Supporting Layers

class Permute(nn.Module):
    def __init__(self, dim0, dim1, dim2):
        super().__init__()
        self.dim0, self.dim1, self.dim2 = dim0, dim1, dim2

    def forward(self, x):
        return x.permute(self.dim0, self.dim1, self.dim2)

class TimeMixingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.Time_Mixing_Sublayer = nn.Sequential(
            # Time Mixing Sublayer
            # Batch_Norm_2D(),
            nn.BatchNorm1d(WINDOW_SIZE),
            nn.Linear(EFF_SENSORS, EFF_SENSORS),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
        )

    def forward(self, x):
        return x + self.Time_Mixing_Sublayer(x)

# class TimeMixingPipe(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.pipeline = nn.ModuleList([TimeMixingLayer() for _ in range(TIME_MIXING_LAYERS)])
#
#     def forward(self, x):
#         for i in range(TIME_MIXING_LAYERS):
#             x = self.pipeline[i](x)
#         return x

class FeatureMixingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.Feature_Mixing_Sublayer = nn.Sequential(
            # Feature Mixing Sublayer
            # Batch_Norm_2D()
            nn.BatchNorm1d(WINDOW_SIZE),
            Permute(0, 2, 1), # X(BATCH_SIZE, EFF_SENSORS, WINDOW_SIZE)

            nn.Linear(WINDOW_SIZE, MLP_FEATURE_OUTSIZE),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),

            nn.Linear(MLP_FEATURE_OUTSIZE, WINDOW_SIZE),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),

            Permute(0, 2, 1) # X(BATCH_SIZE, WINDOW_SIZE, EFF_SENSORS)
        )

    def forward(self, x):
        return x + self.Feature_Mixing_Sublayer(x)

# class FeatureMixingPipe(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.pipeline = nn.ModuleList([FeatureMixingLayer() for _ in range(FEATURE_MIXING_LAYER)])
#
#     def forward(self, x):
#         for i in range(FEATURE_MIXING_LAYER):
#             x = self.pipeline[i](x)
#         return x
class MixingPipe(nn.Module):
    def __init__(self):
        super().__init__()
        self.pipeline = nn.ModuleList()
        for _ in range(MIXING_LAYERS):
            self.pipeline.append(FeatureMixingLayer())
            self.pipeline.append(TimeMixingLayer())

    def forward(self, x):
        for i in range(MIXING_LAYERS):
            x = self.pipeline[2*i](x)
            x = self.pipeline[2*i+1](x)
        return x


# Main Model
class TS_Mixer(nn.Module):
    def __init__(self):
        super(TS_Mixer, self).__init__()

        # self.Time_Mixing_Sublayer = nn.Sequential(
        #     # Time Mixing Sublayer
        #     # Batch_Norm_2D(),
        #     nn.BatchNorm1d(WINDOW_SIZE),
        #     nn.Linear(EFF_SENSORS, EFF_SENSORS),
        #     nn.ReLU(),
        #     nn.Dropout(p=DROPOUT),
        # )
        #
        # self.Feature_Mixing_Sublayer = nn.Sequential(
        #     # Feature Mixing Sublayer
        #     # Batch_Norm_2D()
        #     nn.BatchNorm1d(WINDOW_SIZE),
        #     Permute(0, 2, 1), # X(BATCH_SIZE, EFF_SENSORS, WINDOW_SIZE)
        #
        #     nn.Linear(WINDOW_SIZE, MLP_FEATURE_OUTSIZE),
        #     nn.ReLU(),
        #     nn.Dropout(p=DROPOUT),
        #
        #     nn.Linear(MLP_FEATURE_OUTSIZE, WINDOW_SIZE),
        #     nn.ReLU(),
        #     nn.Dropout(p=DROPOUT),
        #
        #     Permute(0, 2, 1) # X(BATCH_SIZE, WINDOW_SIZE, EFF_SENSORS)
        # )
        self.mixing_pipe = MixingPipe()

        self.Temporal_Projection = nn.Sequential(
            Permute(0, 2, 1), # X(BATCH_SIZE, EFF_SENSORS, WINDOW_SIZE)

            nn.Linear(WINDOW_SIZE, 16),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),

            nn.Linear(16, 1),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),

            Permute(0, 2, 1), # X(BATCH_SIZE, WINDOW_SIZE=1, EFF_SENSORS)

            nn.Linear(EFF_SENSORS, 16),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),

            nn.Linear(16, 1),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
        )

    def forward(self, X): # X(BATCH_SIZE, WINDOW_SIZE, EFF_SENSORS)

        # for i in range(4):
        #     # ResNet
        #     X = X + self.Time_Mixing_Sublayer(X)
        #     X = X + self.Feature_Mixing_Sublayer(X)
        X = self.mixing_pipe(X)
        out = self.Temporal_Projection(X).view(-1, 1)

        return out