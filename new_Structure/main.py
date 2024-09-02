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
        pth_to_load = os.path.join('.', 'model_backup', self.arg.dataset)

        if len(os.listdir(pth_to_load)) == 1:
            file = os.listdir(pth_to_load)[0]
        else:
            raise ValueError("No model to load")

        file_path = os.path.join(pth_to_load, file)

        self.net.load_state_dict(torch.load(file_path))
        print(f"Model loaded from {file_path}")

    def DrawTrainEngineWithInputWindowSize(self, window_size):
        train_dataloader = self.data.getTrainEngineDataloader(
            window_size=window_size,
            batch_size=self.arg.batch_size,
            engine_num=self.arg.train_max_rul_dict['id'],
            memory_pinned=self.arg.memory_pinned
        )

        outputs, targets = torch.full((self.arg.train_max_rul_dict['RUL'],), torch.inf), torch.full((self.arg.train_max_rul_dict['RUL'],), torch.inf)
        i = window_size-1 # REMIND: starting not from zero for ploting
        for data, target in train_dataloader:
            data, target = data.to(self.arg.device), target.to(self.arg.device)
            output = self.net(data)

            for output_item, target_item in zip(output, target):
                outputs[i] = output_item.cpu().detach()
                targets[i] = target_item.cpu().detach()
                i += 1

            del output, target

        RMSE = pow(self.loss_function(outputs, targets).item(), 0.5)*self.arg.max_rul
        score = self.score_function(outputs, targets).item()
        outputs_np = outputs.numpy()*self.arg.max_rul
        targets_np = targets.numpy()*self.arg.max_rul

        # data = self.data.train_data.get_


        self.line_visualize(outputs_np, targets_np, RMSE, score)


    #             train_dataloader = self.data.getTrainDataloader(
    #                 window_size=window_size,
    #                 batch_size=self.arg.batch_size,
    #                 memory_pinned=self.arg.memory_pinned
    #             )
    #             for data, target in train_dataloader:
    #                 data, target = data.to(self.arg.device), target.to(self.arg.device)
    #                 output = self.net(data)
    #
    #                 self.optimizer.zero_grad()
    #                 loss = self.loss_function(output, target)
    #                 train_loss += loss.item()
    #                 loss.backward()
    #                 self.optimizer.step()
    #
    #                 del data, target, output
    #
    #
    #             with torch.no_grad():
    #                 self.net.eval()
    #                 test_dataloader = self.data.getTestDataloader(
    #                     batch_size=1,
    #                     memory_pinned=self.arg.memory_pinned
    #                 )
    #                 i = 0
    #                 outputs, targets = torch.zeros(self.data.test_engine_num), torch.zeros(self.data.test_engine_num)
    #                 for data, target in test_dataloader:
    #                     data, target = data.to(self.arg.device), target.to(self.arg.device)
    #                     output = self.net(data)
    #
    #                     outputs[i], targets[i] = output.cpu().detach(), target.cpu().detach()
    #                     i += 1
    #
    #                     del data, target, output
    #
    #             test_RMSE   = pow(self.loss_function(outputs, targets).item(), 0.5)
    #             test_score  = self.score_function(outputs, targets).item()
    #             outputs_cpu = outputs.numpy()*self.arg.max_rul
    #             targets_cpu = targets.numpy()*self.arg.max_rul
    #             test_RMSE   = test_RMSE*self.arg.max_rul
    #
    #             if test_RMSE < test_RMSE_best:
    #                 test_RMSE_best = test_RMSE
    #                 print('Epoch: {:03d}, '
    #                       'Train Loss: {:.4f}, '
    #                       'Test RMSE: {:.4f}, '
    #                       'Test Score: {:.4f}, '
    #                       'training Window Size: {}'.format(epoch, train_loss, test_RMSE, test_score, window_size))
    #                 self.visualize(outputs_cpu, targets_cpu, test_RMSE, test_score)
    #                 self.save_best_model_param(test_RMSE)
    #
    #     return float(test_RMSE_best)


    def score_function(self, predicts, reals):
        score = 0
        num = predicts.size(0)
        for i in range(num):
            target = torch.tensor(0) if reals[i] == torch.inf else reals[i]
            predict = torch.tensor(0) if predicts[i] == torch.inf else predicts[i]

            if target > predict:
                score = score+ (torch.exp((target*self.arg.max_rul-predict*self.arg.max_rul)/13)-1)

            elif target<= predict:
                score = score + (torch.exp((predict*self.arg.max_rul-target*self.arg.max_rul)/10)-1)

        return score

    def line_visualize(self, result, y_test, rmse, score):
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

        ax.plot(
            true_rul,
            color='blue',
            label='Actual RUL',
            linewidth=4,
            zorder=0,
        )
        ax.plot(
            pred_rul,
            color='red',
            label='Predicted RUL RMSE = {} Score = {})'.format(round(rmse, 3), int(score)),
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
            arguments.window_size_tuple = (arguments.accept_window, 70, 80, 90, 100, 110,)
            arguments.batch_size    = 1
            arguments.train_max_rul_dict = {
                'id' : 69,
                'RUL': 362,
            }

        case 2:
            arguments.accept_window = 50
            arguments.window_size_tuple = (arguments.accept_window, 60, 70, 80, 90, 100, 110, 120,)
            arguments.batch_size    = 1
            arguments.train_max_rul_dict = {
                'id' : 112,
                'RUL': 378,
            }

        case 3:
            arguments.accept_window = 50
            arguments.window_size_tuple = (arguments.accept_window, 60, 70, 80, 90, 100, )
            arguments.batch_size    = 1
            arguments.train_max_rul_dict = {
                'id' : 55,
                'RUL': 525,
            }

        case 4:
            arguments.accept_window = 50
            # arguments.window_size_tuple = (arguments.accept_window, 60, 70, 80, 90, 100, 110, 120,)
            arguments.window_size_tuple = (arguments.accept_window, 60, 70, 80)
            arguments.batch_size    = 1
            arguments.train_max_rul_dict = {
                'id' :118,
                'RUL':543,
            }

        case _:
            raise ValueError("Invalid dataset choice")

    return arguments

def main() -> None:
    # REMIND: model must have its name attribute
    args = args_config(
        dataset_choice=,
    )

    model = LSTM_pTSMixer_GA(
        sensors=14, e_layers=16,
        t_model=48, c_model=36,
        lstm_layer_num=2,
        seq_len=args.accept_window, dropout=0.2, accept_window=args.accept_window)

    args.model_name = model.name

    instance = Process(args, model)
    instance.DrawTrainEngineWithInputWindowSize(window_size=60)


if __name__ == '__main__':
    with torch.no_grad():
        main()
