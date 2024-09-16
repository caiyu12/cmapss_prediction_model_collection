from test_model import *
from data_process import CMAPSS_Data_Process

from argparse import Namespace
import torch
import numpy
import pandas
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

class Process():
    def __init__(self, arg : Namespace, model) -> None:
        self.arg = arg
        self.data = CMAPSS_Data_Process(self.arg)

        self.net = model.to(arg.device)
        self.loss_function = torch.nn.MSELoss()

        self.load_model()

    def load_model(self):
        self.net.eval()
        pth_to_load = os.path.join('.', 'model_backup', self.arg.dataset)

        if len(os.listdir(pth_to_load)) == 1:
            file = os.listdir(pth_to_load)[0]
        else:
            raise ValueError("No model to load")

        file_path = os.path.join(pth_to_load, file)

        self.net.load_state_dict(torch.load(file_path))
        print(f"Model loaded from {file_path}")

    def DrawTrainEngineWithInputWindowSize(self):
        assert self.arg.window_size >= self.arg.accept_window, 'Window size must be greater than or equal to accept window'
        data_gp = self.data.train_data
        targ_gp = self.data.train_target
        assert data_gp.ngroups == targ_gp.ngroups, "Data and target group number mismatch"
        assert self.arg.engine_choice <= data_gp.ngroups, "Engine choice higher than valid range"
        assert self.arg.engine_choice > 0, "Engine choice lower than valid range"

        data = data_gp.get_group(self.arg.engine_choice).iloc[:, 1:].to_numpy()
        targ = targ_gp.get_group(self.arg.engine_choice).iloc[:, 1:].to_numpy().reshape(-1)
        assert self.arg.window_size <= data.shape[0], "Window size higher than valid range"


        iter_length = data.shape[0] - self.arg.window_size + 1
        full_length = data.shape[0]
        outputs = torch.zeros(iter_length)
        targets = torch.tensor(targ[self.arg.window_size-1:])

        for i in range(iter_length):
            input = torch.Tensor(data[i:i+self.arg.window_size, :]).to(self.arg.device).unsqueeze(0)
            output = self.net(input)
            outputs[i] = output.detach().cpu()

            del output

        RMSE = pow(self.loss_function(outputs, targets).item(), 0.5)*self.arg.max_rul
        score = self.score_function(outputs, targets).item()
        outputs_np = outputs.numpy()*self.arg.max_rul
        targets_np = targets.numpy()*self.arg.max_rul
        targfull_np = targ*self.arg.max_rul

        # data = self.data.train_data.get_


        self.line_visualize(
            outputs_np, targets_np, targfull_np,
            RMSE, score,
            self.arg.engine_choice, self.arg.window_size
        )

    def Test(self):
        epoch = 1

        test_dataloader = self.data.getTestDataloader(
            batch_size=1,
            memory_pinned=self.arg.memory_pinned
        )
        i = 0
        outputs, targets = torch.zeros(self.data.test_engine_num), torch.zeros(self.data.test_engine_num)
        for data, target in test_dataloader:
            data, target = data.to(self.arg.device), target.to(self.arg.device)
            output = self.net(data)

            outputs[i], targets[i] = output.cpu().detach(), target.cpu().detach()
            i += 1

            del data, target, output

        test_RMSE   = pow(self.loss_function(outputs, targets).item(), 0.5)
        test_score  = self.score_function(outputs, targets).item()
        outputs_cpu = outputs.numpy()*self.arg.max_rul
        targets_cpu = targets.numpy()*self.arg.max_rul
        test_RMSE   = test_RMSE*self.arg.max_rul

        print('Epoch: {:03d}, '
              'Test RMSE: {:.4f}, '
              'Test Score: {:.4f}, '.format(epoch, test_RMSE, test_score,))
        self.scatter_visualize(outputs_cpu, targets_cpu, test_RMSE, test_score)


    def score_function(self, predicts, reals):
        score = 0
        num = predicts.size(0)
        for i in range(num):
            target = reals[i]
            predict = predicts[i]

            if target > predict:
                score = score+ (torch.exp((target*self.arg.max_rul-predict*self.arg.max_rul)/13)-1)

            elif target<= predict:
                score = score + (torch.exp((predict*self.arg.max_rul-target*self.arg.max_rul)/10)-1)

        return score

    def line_visualize(self, result, y_test, y_test_full, rmse, score, choice, window):
        length_full = len(y_test_full)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            range(length_full),
            y_test_full,
            color='blue',
            label='Actual RUL',
            linewidth=4,
            zorder=0,
        )
        ax.plot(
            range(window-1, length_full),
            result,
            color='red',
            label='Predicted RUL RMSE = {} Score = {})'.format(round(rmse, 3), int(score)),
        )
        ax.set_title('Remaining Useful Life Prediction--{} on {}, Engine #{}'.format(
            self.arg.model_name, self.arg.dataset, choice
            )
        )
        ax.legend()
        plt.show()

    def scatter_visualize(self, result, y_test, rmse, score):
        """

        :param result: RUL prediction results
        :param y_test: true RUL of testing set
        :param num_test: number of samples
        :param rmse: RMSE of prediction results
        """
        result = numpy.array(result, dtype=object).reshape(-1, 1)
        num_test = len(result)
        y_test = pandas.DataFrame(y_test, columns=['RUL'])
        result = y_test.join(pandas.DataFrame(result))
        result = result.sort_values('RUL', ascending=False)
        rmse = float(rmse)

        # the true remaining useful life of the testing samples
        true_rul = result.iloc[:, 0].to_numpy()
        # the predicted remaining useful life of the testing samples
        pred_rul = result.iloc[:, 1].to_numpy()

        fig, ax = plt.subplots(figsize=(10, 6))
        # plt.axvline(x=num_test, c='r', linestyle='--')  # size of the training set

        ax.plot(
            true_rul,
            color='blue',
            label='Actual RUL',
            linewidth=4,
            zorder=0,
        )  # actual plot

        # err outside -10 to 10 will be printed as prue color
        max_err_clip =  20
        min_err_clip = -20
        err = pred_rul - true_rul
        cliped_err = err.clip(min_err_clip, max_err_clip)
        cliped_normed_err = (cliped_err - min_err_clip) / (max_err_clip - min_err_clip)

        colors = ['green', 'white', 'red']
        nodes = [0, 0.5, 1]  # 0: far below, 0.5: same, 1: far above
        color_map = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))
        sc = ax.scatter(
            x=range(len(pred_rul)),
            y=pred_rul,
            c=cliped_normed_err,
            cmap=color_map,
            edgecolor='k',
            s=100,
            label='Predicted RUL RMSE = {} Score = {})'.format(round(rmse, 3), int(score)),
            zorder=1,
        )  # predicted plot

        # add color bar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('ERROR (Predicted RUL - Actual RUL)')
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels([f'{min_err_clip} or below', '0', f'+{max_err_clip} or above'])

        ax.set_title('Remaining Useful Life Prediction--{} on {}'.format(self.arg.model_name, self.arg.dataset))
        ax.legend()

        ax.set_xlabel("Samples")
        ax.set_ylabel("Remaining Useful Life")
        plt.show()


def args_config(dataset_choice : int) -> Namespace:
    arguments = Namespace(
        directory = './',
        dataset   = 'FD00{}'.format(dataset_choice),
        epoch     = 10,
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_rul   = 125,
        learning_rate = 0.001,

        memory_pinned = True,
        # REMIND: place model hyperparameters here
    )
    match dataset_choice:
        case 1:
            arguments.accept_window = 60
            arguments.train_max_rul_dict = {
                'id' : 69,
                'RUL': 362,
            }
            arguments.engine_choice = 69
            arguments.window_size = 60

        case 2:
            arguments.accept_window = 50
            arguments.train_max_rul_dict = {
                'id' : 112,
                'RUL': 378,
            }
            arguments.engine_choice = 1
            arguments.window_size = 50

        case 3:
            arguments.accept_window = 50
            arguments.train_max_rul_dict = {
                'id' : 55,
                'RUL': 525,
            }
            arguments.engine_choice = 55 # 9
            arguments.window_size = 50

        case 4:
            arguments.accept_window = 40
            arguments.train_max_rul_dict = {
                'id' :118,
                'RUL':543,
            }
            arguments.engine_choice = 11
            arguments.window_size = 40

        case _:
            raise ValueError("Invalid dataset choice")

    return arguments

def main() -> None:
    # REMIND: model must have its name attribute
    args = args_config(
        dataset_choice=4,
    )

    model = LSTM_pTSMixer_GA(
        sensors=14, e_layers=8,
        t_model=36, c_model=36,
        lstm_layer_num=8,
        seq_len=args.accept_window, dropout=0.2, accept_window=args.accept_window)

    args.model_name = model.name

    instance = Process(args, model)
    # instance.Test()
    instance.DrawTrainEngineWithInputWindowSize()


if __name__ == '__main__':
    with torch.no_grad():
        main()
