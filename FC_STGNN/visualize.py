"""Prediction result visualization"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

now = datetime.datetime.now()
DIR_NAME = now.strftime('%H-%M-%S %m-%d, %Y')

def visualize(result, y_test, num_test, rmse, data_sub):
    """

    :param result: RUL prediction results
    :param y_test: true RUL of testing set
    :param num_test: number of samples
    :param rmse: RMSE of prediction results
    """
    result = np.array(result, dtype=object).reshape(100, -1)
    result = y_test.join(pd.DataFrame(result))
    result = result.sort_values('RUL', ascending=False)

    # the true remaining useful life of the testing samples
    true_rul = result.iloc[:, 0].to_numpy()
    # the predicted remaining useful life of the testing samples
    pred_rul = result.iloc[:, 1].to_numpy()

    plt.figure(figsize=(10, 6))  # plotting
    plt.axvline(x=num_test, c='r', linestyle='--')  # size of the training set

    plt.plot(true_rul, label='Actual RUL')  # actual plot
    plt.plot(pred_rul, label='Predicted RUL (RMSE = {})'.format(round(rmse, 3)))  # predicted plot
    plt.title('Remaining Useful Life Prediction')
    plt.legend()

    plt.xlabel("Samples")
    plt.ylabel("Remaining Useful Life")

    plt.text(
        x =  0,
        y =  20,

        s='TEST_DATASET=FD00{}'
        .format(data_sub),

        bbox={
            'facecolor': 'none',
            'edgecolor': 'black',
            'boxstyle': 'round'
        }
    )


    plt.savefig('./_trials/{}/{} RUL Prediction.png'.format(DIR_NAME, round(rmse, 3)))
    plt.show()
