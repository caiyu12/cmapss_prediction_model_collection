from test_model import *
from data_process import CMAPSS_Data_Process

from argparse import Namespace
import torch
import numpy
import pandas
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

class Train():
    def __init__(self, arg : Namespace, model) -> None:
        self.arg = arg
        self.data = CMAPSS_Data_Process(self.arg)

        self.net = model.to(arg.device)
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=arg.learning_rate)

    def Train_Test(self)->float:
        test_RMSE_best = numpy.inf
        for epoch in range(1, self.arg.epoch + 1):
            for window_size in self.arg.window_size_tuple:

                train_loss = 0
                test_RMSE  = 0
                test_score = 0

                self.net.train()
                train_dataloader = self.data.getTrainDataloader(
                    window_size=window_size,
                    batch_size=self.arg.batch_size,
                    memory_pinned=self.arg.memory_pinned
                )
                for data, target in train_dataloader:
                    data, target = data.to(self.arg.device), target.to(self.arg.device)
                    output = self.net(data)

                    self.optimizer.zero_grad()
                    loss = self.loss_function(output, target)
                    train_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()

                    del data, target, output


                with torch.no_grad():
                    self.net.eval()
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

                if test_RMSE < test_RMSE_best:
                    test_RMSE_best = test_RMSE
                    print('Epoch: {:03d}, '
                          'Train Loss: {:.4f}, '
                          'Test RMSE: {:.4f}, '
                          'Test Score: {:.4f}, '
                          'training Window Size: {}'.format(epoch, train_loss, test_RMSE, test_score, window_size))
                    self.visualize(outputs_cpu, targets_cpu, test_RMSE, test_score)
                    self.save_best_model_param(test_RMSE)

        return float(test_RMSE_best)


    def score_function(self, predicted, real):
        score = 0
        num = predicted.size(0)
        for i in range(num):

            if real[i] > predicted[i]:
                score = score+ (torch.exp((real[i]*self.arg.max_rul-predicted[i]*self.arg.max_rul)/13)-1)

            elif real[i]<= predicted[i]:
                score = score + (torch.exp((predicted[i]*self.arg.max_rul-real[i]*self.arg.max_rul)/10)-1)

        return score

    def visualize(self, result, y_test, rmse, score):
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

    def save_best_model_param(self, new_criterion_value : float):
        new_criterion_value = round(new_criterion_value, 3)

        file_dir = os.path.join('./param_model', self.arg.dataset)
        file_list = os.listdir(file_dir)

        new_file = os.path.join(file_dir, str(new_criterion_value) + '_' + self.arg.model_name + '.pth')

        file_num  = len(file_list)
        match file_num:
            case 0:
                torch.save(self.net.state_dict(), new_file)

            case 1:
                old_file_name = file_list[0]
                old_file = os.path.join(file_dir, old_file_name)
                old_criterion_value = float(old_file_name.split('_')[0])

                if new_criterion_value < old_criterion_value:
                    os.remove(old_file)
                    torch.save(self.net.state_dict(), new_file)
                else:
                    pass

            case _:
                IOError('param_model directory structure error, please check it.')


def args_config(dataset_choice : int) -> Namespace:
    arguments = Namespace(
        directory = './',
        dataset   = 'FD00{}'.format(dataset_choice),
        epoch     = 10,
        device    = torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
        max_rul   = 125,
        learning_rate = 0.001,

        memory_pinned = True,
        # REMIND: place model hyperparameters here
    )
    match dataset_choice:
        case 1:
            arguments.accept_window = 60
            arguments.window_size_tuple = (arguments.accept_window, 70, 80, 90, 100, 110,)
            arguments.batch_size    = 100

        case 2:
            arguments.accept_window = 50
            arguments.window_size_tuple = (arguments.accept_window, 60, 70, 80, 90, 100, 110, 120,)
            arguments.batch_size    = 100

        case 3:
            arguments.accept_window = 50
            arguments.window_size_tuple = (arguments.accept_window, 60, 70, 80, 90, 100, )
            arguments.batch_size    = 100

        case 4:
            arguments.accept_window = 50
            # arguments.window_size_tuple = (arguments.accept_window, 60, 70, 80, 90, 100, 110, 120,)
            arguments.window_size_tuple = (arguments.accept_window, 60, 70, 80)
            arguments.batch_size    = 100

        case _:
            raise ValueError("Invalid dataset choice")

    return arguments

def main() -> None:
    # REMIND: model must have its name attribute
    args = args_config(
        dataset_choice=4,
    )
    # model = TSMixer(sensors=14, e_layers=8, d_model=36, seq_len=args.accept_window, pred_len=1, dropout=0.2)
    # model = parallel_TSMixer(sensors=14, e_layers=16, d_model=36, seq_len=args.accept_window, pred_len=1, dropout=0.2)
    # model = LSTM_TSMixer(
    #     sensors=14,
    #     e_layers=16, # 8  # 16
    #     d_model=36, # 36 # 48
    #     seq_len=args.accept_window, pred_len=1, dropout=0.2, accept_window=args.accept_window
    # )

    model = LSTM_pTSMixer_GA(
        sensors=14, e_layers=16,
        t_model=48, c_model=36,
        lstm_layer_num=2,
        seq_len=args.accept_window, dropout=0.2, accept_window=args.accept_window)

    # model = ENCODER_LSTM_TSMixer(sensors=14, e_layers=8, d_model=36, seq_len=args.accept_window, pred_len=1, dropout=0.2, accept_window=args.accept_window)
    args.model_name = model.name

    train = Train(args, model)
    rmse = train.Train_Test()

if __name__ == '__main__':
    for i in range(20):
        main()
