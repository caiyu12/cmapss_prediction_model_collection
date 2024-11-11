import torch.nn as nn

'''
params：seq_len:input sequence length
pred_len:prediction sequence length
d_model:embedding dimension
n_heads:head number
d_ff:feed forward dimension
dropout:dropout rate
'''
class ResBlock(nn.Module):
    def __init__(self, sensors, seq_len, d_model, dropout):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(d_model, seq_len),
            # nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(sensors, d_model),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(d_model, sensors),
            # nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x

# class Mixup(nn.Module):
#     def __init__(self, sensors, e_layers, seq_len, d_model, dropout):
#         super(Mixup, self).__init__()
#         self.mixup = nn.ModuleList(
#             [ResBlock(sensors, seq_len, d_model, dropout)
#              for _ in range(e_layers)]
#         )

class LSTM_TSMixer(nn.Module):
    def __init__(self, sensors, e_layers, d_model, seq_len, pred_len, dropout, accept_window):
        super(LSTM_TSMixer, self).__init__()
        self.name = 'LSTM_TSMixer'
        self.layer = e_layers
        self.model = nn.ModuleList(
            [ResBlock(sensors, seq_len, d_model, dropout)
                                    for _ in range(e_layers)]
        )
        self.pred_len = pred_len
        # self.projection = nn.Linear(seq_len, pred_len)
        # self.squeeze = nn.Linear(sensors, pred_len)
        self.projection = nn.Sequential(
            nn.Linear(seq_len, pred_len),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
        self.squeeze    = nn.Sequential(
            nn.Linear(sensors, pred_len),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(input_size=sensors, hidden_size=sensors, num_layers=1, batch_first=True)
        self.accept_window = accept_window

    def forecast(self, x_enc):
        x_enc, _ = self.lstm(x_enc)
        x_enc = x_enc[:, -self.accept_window:, :].contiguous()

        # x: [B, L, D]
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        enc_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
        enc_out_2d = enc_out.view(-1, 14)
        enc_output = self.squeeze(enc_out_2d)


        return enc_output

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:]  # [B, L, D]
