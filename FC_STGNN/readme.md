# Python implementation of C-MAPSS Dataset RUL Prediction

## Intro
This repository contains the python implementation of the C-MAPSS dataset RUL prediction. 
The skeleton code for data preprocessing is derived from [Fully-Connected Spatial-Temporal Graph Neural Network for Multivariate Time-Series Data](https://github.com/Frank-Wang-oss/FCSTGNN). 

## Experiments
We have tested series of models on the C-MAPSS dataset including:
[FC-STGNN](https://arxiv.org/pdf/2309.05305.pdf), [TSMixer](https://arxiv.org/abs/2303.06053.pdf), Encoder, LSTM, T-CNN, and cvCNN(self-developed)
Comparison Model: GA-TCNN, 
### test result:

| Model/Dataset |      FD001       |      FD002       |      FD003       |      FD004       |
|---------------|:----------------:|:----------------:|:----------------:|:----------------:|
| FC-STGNN      |      11.62       |      13.04       |      11.52       |      13.52       |
| TSMixer       |      11.93       |      12.97       |      11.56       |      13.80       |
| Encoder       |                  |                  |                  |                  |
| LSTM          |      19.79       |      19.09       |      19.22       |      21.94       |
| TCNN          | 13.58(on 3060lt) | 14.90(on 3060lt) | 14.40(on 3060lt) | 15.52(on 3060lt) |
| cvCNN         |                  |                  |                  |                  |

## C-MAPSS Dataset

Access the dataset from [here](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) and put them into directory 'CMAPSSData'.

For running the experiments on C-MAPSS, directly run main_RUL.py
