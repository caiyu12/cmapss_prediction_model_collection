# Python implementation of C-MAPSS Dataset RUL Prediction

This repository contains the python implementation of the C-MAPSS dataset RUL prediction. 
The skeleton code for data preprocessing is derived from [Fully-Connected Spatial-Temporal Graph Neural Network for Multivariate Time-Series Data](https://github.com/Frank-Wang-oss/FCSTGNN). 

We have tested series of models on the C-MAPSS dataset including:
[FC-STGNN](https://arxiv.org/pdf/2309.05305.pdf), [TSMixer](https://arxiv.org/abs/2303.06053.pdf), Encoder, LSTM, T-CNN, and cvCNN(self-developed)
Comparison Model: GA-TCNN, 
test result:

| Model/Dataset |      FD001       |      FD002       |      FD003       |      FD004       |
|---------------|:----------------:|:----------------:|:----------------:|:----------------:|
| FC-STGNN      |      11.62       |      13.04       |      11.52       |      13.52       |
| TSMixer       |                  |      12.97       |                  |      13.80       |
| Encoder       |                  |                  |                  |                  |
| LSTM          |                  |                  |                  |                  |
| TCNN          | 13.58(on 3060lt) | 14.90(on 3060lt) | 14.40(on 3060lt) | 15.52(on 3060lt) |
| cvCNN         |                  |                  |                  |                  |
