import numpy as np

from test_model import *
from data_process import CMAPSS_Data_Process

from argparse import Namespace
import torch
import numpy
import pandas
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm, gaussian_kde, binomtest
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

        self.net.load_state_dict(torch.load(file_path, map_location=self.arg.device))
        print(f"Model loaded from {file_path}")

    def TrainEngineWithNOandInputWindowSize(
            self,
            numero: int,
            window: int
    ) -> (numpy.ndarray, numpy.ndarray, numpy.ndarray):
        assert window >= self.arg.accept_window, 'Window size must be greater or equal to accept window'
        data_gp = self.data.train_data
        targ_gp = self.data.train_target
        assert data_gp.ngroups == targ_gp.ngroups, "Data and target group number mismatch"
        assert numero <= data_gp.ngroups, "Engine choice higher than valid range"
        assert numero > 0, "Engine choice lower than valid range"
        assert window <= data_gp.get_group(numero).shape[0], "Window size higher than valid range"

        data = data_gp.get_group(numero).iloc[:, 1:].to_numpy()
        targ = targ_gp.get_group(numero).iloc[:, 1:].to_numpy().reshape(-1)

        iter_length = data.shape[0] - window + 1
        full_length = data.shape[0]
        outputs = torch.zeros(iter_length)
        targets = torch.tensor(targ[window-1:])

        for i in range(iter_length):
            input = torch.Tensor(data[i:i+window, :]).to(self.arg.device).unsqueeze(0)
            output = self.net(input)
            outputs[i] = output.detach().cpu()

            del output

        RMSE = pow(self.loss_function(outputs, targets).item(), 0.5)*self.arg.max_rul
        score = self.score_function(outputs, targets).item()
        outputs_np = outputs.numpy()*self.arg.max_rul
        targets_np = targets.numpy()*self.arg.max_rul
        targfull_np = targ*self.arg.max_rul

        return outputs_np, targets_np, targfull_np


    def DrawTrainEnginePredOnArgWin(self):
        outputs, targets, targfull = self.TrainEngineWithNOandInputWindowSize(
            self.arg.engine_choice, self.arg.window_size
        )
        RMSE = pow(self.loss_function(torch.tensor(outputs), torch.tensor(targets)).item(), 0.5)
        score = self.score_function(torch.tensor(outputs), torch.tensor(targets)).item()


        self.line_visualize(
            outputs, targets, targfull,
            RMSE, score,
            self.arg.engine_choice, self.arg.window_size
        )

    def RMSE60OfTrainEngineOnArgDataset(self):
        engine_num = self.data.train_engine_num
        metric_length = 60

        RMSE_origins_np     = np.zeros(engine_num)
        RMSE_differences_np = np.zeros(engine_num)
        for i in range(1, engine_num+1):
            outputs_origin, targets_origin, targfull_origin = self.TrainEngineWithNOandInputWindowSize(
                i, self.arg.accept_window
            )
            outputs_LargWin, targets_LargWin, targfull_LargWin = self.TrainEngineWithNOandInputWindowSize(
                i, self.arg.accept_window + 5
            )

            RMSE_origin = pow(self.loss_function(
                torch.tensor(outputs_origin[-metric_length:]), torch.tensor(targets_origin[-metric_length:])
            ).item(), 0.5)
            RMSE_LargWin = pow(self.loss_function(
                torch.tensor(outputs_LargWin[-metric_length:]), torch.tensor(targets_LargWin[-metric_length:])
            ).item(), 0.5)

            RMSE_difference = RMSE_origin - RMSE_LargWin

            RMSE_origins_np[i-1] = RMSE_origin
            RMSE_differences_np[i-1] = RMSE_difference
            print('NO:{} RMSE_difference:{}'.format(i, RMSE_difference))

        numpy.save('./RMSEarray/{}_RMSE_origin.npy'.format(self.arg.dataset), RMSE_origins_np)
        numpy.save('./RMSEarray/{}_RMSE_difference.npy'.format(self.arg.dataset), RMSE_differences_np)



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
        test_score  = self.score_function(outputs*self.arg.max_rul, targets*self.arg.max_rul).item()
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
                score = score+ (torch.exp((target-predict)/13)-1)

            elif target<= predict:
                score = score + (torch.exp((predict-target)/10)-1)

        return score


    def sign_test(self):
        data = np.load('./RMSEarray/{}_RMSE_difference.npy'.format(self.arg.dataset))
        count = sum([1 for i in data if i > 0]) # REMIND: Here we count the number of positive values, which means improvement in RMSE

        # REMIND: Here we use binomtest to test the Null Hypothesis that the probability of improvement is 0.5,
        #  and the alternative hypothesis is that the probability of improvement is greater than 0.5
        result = binomtest(count, len(data), p=0.5, alternative='greater')
        print(result)


    def bell_visualize(self):
        data = np.load('./RMSEarray/{}_RMSE_difference.npy'.format(self.arg.dataset))

        mean = np.mean(data)
        std_dev = np.std(data)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax2 = ax1.twinx()

        xmin, xmax = mean - 3*std_dev, mean + 3*std_dev
        x = np.linspace(xmin, xmax, 1000)
        p = norm.pdf(x, mean, std_dev)

        # 绘制整数柱状图
        hist, bin_edges = np.histogram(data, bins=range(int(xmin), int(xmax)+2))
        ax1.bar(
            bin_edges[:-1],
            hist,
            width=np.diff(bin_edges),
            color='skyblue', edgecolor='black',
            alpha=1,
            label='Bars',
            zorder=1
        )
        ax1.set_xlabel('improvement', color='k', fontname=self.arg.fontname, fontsize=self.arg.axis_fontsize)
        ax1.set_ylabel('frequency', color='k',   fontname=self.arg.fontname, fontsize=self.arg.axis_fontsize)
        ax1.tick_params(axis='y', labelcolor='k')
        # 绘制正态分布的 bell curve
        ax2.plot(
            x,
            p,
            'r',
            linewidth=4,
            label='Bell Curve',
            zorder=2
        )

        # 绘制KDE
        kde = gaussian_kde(data)
        kde_values = kde(x)
        ax2.plot(
            x,
            kde_values,
            'g',
            linewidth=4,
            label='KDE',
        )
        ax2.set_ylabel('probability Density', color='k', fontname=self.arg.fontname, fontsize=self.arg.axis_fontsize)
        ax2.tick_params(axis='y', labelcolor='k')

        # 添加标题
        plt.title(
            'RMSE reduction on {} train dataset'.format(self.arg.dataset),
            fontname=self.arg.fontname,
            fontsize=self.arg.title_fontsize
        )
        # 添加图例
        ax1.legend(loc='upper left', fontname=self.arg.fontname, fontsize=self.arg.legend_fontsize)
        ax2.legend(loc='upper right', fontname=self.arg.fontname, fontsize=self.arg.legend_fontsize)
        # 显示图表
        fig.tight_layout()
        plt.show()

    def bar_visualize(self, size, RMSE_real, RMSE_diff):
        width = 0.35

        x_labels = numpy.arange(1, size+1)
        fig, ax = plt.subplots(figsize=(10, 6))

        bars_real = ax.bar(x_labels - width/2, RMSE_real, color='b', label='RMSE origin')
        bars_diff = ax.bar(x_labels + width/2, RMSE_diff, color='r', label='RMSE difference')
        for i, bar in enumerate(bars_real):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{x_labels[i]}',
            ha='center', va='bottom' if bar.get_height() >= 0 else 'top', fontname=self.arg.fontname, fontsize=self.arg.legend_fontsize)

        for i, bar in enumerate(bars_diff):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{x_labels[i]}',
            ha='center', va='bottom' if bar.get_height() >= 0 else 'top', fontname=self.arg.fontname, fontsize=self.arg.legend_fontsize)

            ax.set_xlabel('Index')
            ax.set_ylabel('RMSE')
            ax.set_title('Bar chart of RMSE60 on {}'.format(self.arg.dataset), fontname=self.arg.fontname, fontsize=self.arg.title_fontsize)
            ax.legend()
            # 显示图表
            # plt.tight_layout()
            plt.show()

    def line_visualize(
            self,
            result : numpy.ndarray, y_test : numpy.ndarray, y_test_full : numpy.ndarray,
            rmse, score, choice, window):
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
        ax.set_title('Remaining Useful Life Prediction--{} on {}, Engine #{}, Window Size={}'.format(
            self.arg.model_name, self.arg.dataset, choice, window
            ), fontname=self.arg.fontname, fontsize=self.arg.title_fontsize
        )
        ax.legend(fontname=self.arg.fontname, fontsize=self.arg.legend_fontsize)
        plt.show()

    def scatter_visualize(self, result, y_test, rmse, score):
        result = np.array(result, dtype=object).reshape(-1, 1)
        num_test = len(result)
        y_test = pandas.DataFrame(y_test, columns=['RUL'])
        result = y_test.join(pandas.DataFrame(result))
        rmse = float(rmse)

        true_rul = result.iloc[:, 0].to_numpy()
        pred_rul = result.iloc[:, 1].to_numpy()

        if self.arg.dataset == 'FD004':
            fig, ax = plt.subplots(figsize=(20, 4))
        else:
            fig, ax = plt.subplots(figsize=(20, 3))
        ax.grid(True)

        ax.plot(true_rul, color='blue', label='Actual RUL', linewidth=1, linestyle='--', zorder=0)
        ax.scatter(x=range(num_test), y=true_rul, c='blue', edgecolor='k', s=100, marker='o', zorder=0)

        max_err_clip = 20
        min_err_clip = -20
        err = pred_rul - true_rul
        clipped_err = err.clip(min_err_clip, max_err_clip)
        clipped_normed_err = (clipped_err - min_err_clip) / (max_err_clip - min_err_clip)

        colors = ['green', 'white', 'red']
        nodes = [0, 0.5, 1]
        color_map = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

        sc = ax.scatter(
            x=range(num_test),
            y=pred_rul,
            c=clipped_normed_err,
            cmap=color_map,
            edgecolor='k',
            s=200,
            label='Predicted RUL RMSE = {} Score = {})'.format(round(rmse, 3), int(score)),
            marker='^',
            zorder=2,
        )

        ax.plot(pred_rul, color=(0.9, 0.7, 0.), linewidth=2, linestyle='-', zorder=1)

        # Add a horizontal color bar with reduced height
        if self.arg.dataset == 'FD004':
            cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.1, pad=0.1)
            cbar.set_label('ERROR (Predicted RUL - Actual RUL)', fontname=self.arg.fontname, fontsize=self.arg.legend_fontsize)
            cbar.ax.tick_params(labelsize=self.arg.legend_fontsize)
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels([f'{min_err_clip} or below', '0', f'+{max_err_clip} or above'])


        ax.set_title('{} on {} test dataset'.format(self.arg.model_name, self.arg.dataset),
                     fontname=self.arg.fontname, fontsize=self.arg.title_fontsize)
        ax.legend(loc='lower left', fontname=self.arg.fontname, fontsize=self.arg.legend_fontsize)


        ax.set_xlabel("Samples", fontname=self.arg.fontname, fontsize=self.arg.axis_fontsize)
        ax.set_ylabel("RUL", fontname=self.arg.fontname, fontsize=self.arg.axis_fontsize)
        fig.tight_layout()
        plt.show()


    #TODO: Implement this function
    def DrawGivenTrainEnginePredOnAutoArgWinForComparison(self, engine, step : int):
        assert step <= 7, 'time-window enlargement step out of range'
        outputs, targets, RMSEs, RMSE60s = list(), list(), list(), list()

        for i in range(step):
            output, target, targfull = self.TrainEngineWithNOandInputWindowSize(
                engine, self.arg.window_size + 5*i
            )
            RMSE = pow(self.loss_function(torch.tensor(output), torch.tensor(target)).item(), 0.5)
            RMSE60 = pow(self.loss_function(torch.tensor(output[-60:]), torch.tensor(target[-60:])).item(), 0.5)
            outputs.append(output)
            targets.append(target)
            RMSEs.append(RMSE)
            RMSE60s.append(RMSE60)

        fig, ax = plt.subplots(figsize=(10, 6))
        line_widths = [2, 3, 5, 8, 14, 18, 22]
        dot_sizes_start = 20
        length_plot = len(targfull)
        colors = [(255/255, 0/255, 0/255),
                  (220/255, 0/255, 0/255),
                  (190/255, 0/255, 0/255),
                  (160/255, 0/255, 0/255),
                  (130/255, 0/255, 0/255),
                  (255/255, 173/255, 173/255),
                  (255/255, 128/255, 0/255),
                  (0/255, 255/255, 0/255),]
        darken_factor = 0.7

        ax.plot(
            range(length_plot),
            targfull,
            color='blue',
            label='Actual RUL',
            linestyle='--',
            linewidth=4,
            zorder=0,
        )

        for i in range(step):
            window_size = self.arg.window_size + 5*i
            ax.plot(
                range(window_size-1, length_plot),
                outputs[i],
                color=colors[i],
                label='Predicted RUL: time-window={}, RMSE={}, RMSE-60={}'.format(
                    window_size, round(RMSEs[i], 3), round(RMSE60s[i], 3),),
                linewidth=line_widths[i],
                zorder=5-i
            )

        ax.set_title('Prediction on {} Train Engine #{}'.format(
            self.arg.dataset, engine),
            fontname=self.arg.fontname,
            fontsize=self.arg.title_fontsize
        )
        ax.legend(loc='lower left', fontname=self.arg.fontname, fontsize=self.arg.legend_fontsize)
        ax.set_xlabel('cycle', fontname=self.arg.fontname, fontsize=self.arg.axis_fontsize)
        ax.set_ylabel('RUL',   fontname=self.arg.fontname, fontsize=self.arg.axis_fontsize)
        fig.tight_layout()
        fig.show()

    def Draw3dData(self, data : numpy.ndarray, zlabel_pad, ztick_pad) -> None:
        n_cycles=data.shape[0]
        n_sensors=data.shape[1]
        time=numpy.arange(n_cycles)
        sensor_names = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(n_sensors):
            ax.plot(time, [i]*n_cycles, data[:, i])

        # Labeling the axes
        ax.xaxis.set_rotate_label(False)
        ax.yaxis.set_rotate_label(False)  # Make y-axis label horizontal
        ax.set_xlabel('Cycles',
                      fontname=self.arg.fontname, fontsize=self.arg.axis_fontsize, rotation=0, labelpad=30)
        ax.set_ylabel('Sensor number',
                      fontname=self.arg.fontname, fontsize=self.arg.axis_fontsize, rotation=0, labelpad=30)
        ax.set_zlabel('Magnitude',
                      fontname=self.arg.fontname, fontsize=self.arg.axis_fontsize,             labelpad=zlabel_pad) # 10 for normalized data

        # Set font size for X-axis tick labels
        ax.set_xticks(np.linspace(0, n_cycles, 5))  # Example tick positions
        ax.set_xticklabels(np.linspace(0, n_cycles, 5, dtype=int),
                           fontname=self.arg.fontname, fontsize=self.arg.legend_fontsize-1)

        # Set font size for Y-axis tick labels
        ax.set_yticks(numpy.arange(n_sensors))
        ax.set_yticklabels(sensor_names,
                           fontname=self.arg.fontname, fontsize=self.arg.legend_fontsize-1)

        # Set font size for Z-axis tick labels
        ax.tick_params(axis='z', pad=ztick_pad) # 0 for normalized data
        ax.set_zticks(np.linspace(0, data.max(), 5))  # Example tick positions
        ax.set_zticklabels(np.linspace(0, data.max(), 5),
                           fontname=self.arg.fontname, fontsize=self.arg.legend_fontsize-1)

        ax.set_box_aspect([3, 2, 1])

        ax.view_init(elev=20, azim=150)
        ax.grid(True)

        # fig.tight_layout()
        fig.subplots_adjust(left=0.001, right=0.999, top=0.999, bottom=0.001)  # Increase these if the box is still too small
        fig.show()


def args_config(dataset_choice : int) -> Namespace:
    arguments = Namespace(
        directory = './',
        dataset   = 'FD00{}'.format(dataset_choice),
        epoch     = 10,
        device    = torch.device("cuda:2" if torch.cuda.is_available() else "cpu"),
        max_rul   = 125,
        learning_rate = 0.001,

        memory_pinned = True,
        # REMIND: place model hyperparameters here
        fontname='Times New Roman',

        title_fontsize=28,
        axis_fontsize=22,
        legend_fontsize=14,
    )
    match dataset_choice:
        case 1:
            arguments.accept_window = 60
            arguments.test_window   = 110
            arguments.train_max_rul_dict = {
                'id' : 69,
                'RUL': 362,
            }
            arguments.engine_choice = 7 # 7 (60, 65, 70), 71 (60, 70)
            arguments.window_size = 60

        case 2:
            arguments.accept_window = 50
            arguments.test_window   = 120
            arguments.train_max_rul_dict = {
                'id' : 112,
                'RUL': 378,
            }
            arguments.engine_choice = 177
            arguments.window_size = 50

        case 3:
            arguments.accept_window = 50
            arguments.test_window   = 100
            arguments.train_max_rul_dict = {
                'id' : 55,
                'RUL': 525,
            }
            arguments.engine_choice = 55 # 9, 77 (50, 60) | 48, 58, 61, [40, 44, 51, 56]
            arguments.window_size = 50

        case 4:
            arguments.accept_window = 40
            arguments.test_window   = 80
            arguments.train_max_rul_dict = {
                'id' :118,
                'RUL':543,
            }
            arguments.engine_choice = 118 # 118 (40~60), 1
            arguments.window_size = 40

        case _:
            raise ValueError("Invalid dataset choice")

    return arguments

def main() -> None:
    # REMIND: model must have its name attribute
    args = args_config(
        dataset_choice=3,
    )

    model = CoSO_pTSMixer_SGA(
        sensors=14, e_layers=8,
        t_model=36, c_model=36, # 48*36 for ReLU
        lstm_layer_num=8,
        seq_len=args.accept_window, dropout=0.2, accept_window=args.accept_window)

    args.model_name = model.name

    instance = Process(args, model)
    # instance.Test()
    # instance.DrawTrainEnginePredOnArgWin()
    # instance.RMSE60OfTrainEngineOnArgDataset()
    # instance.bell_visualize()
    # instance.sign_test()

    # instance.DrawGivenTrainEnginePredOnAutoArgWinForComparison(58, 5)
    # instance.DrawGivenTrainEnginePredOnAutoArgWinForComparison(args.engine_choice, 5)
    # for i in range(1, instance.data.train_engine_num + 1):
    #     instance.DrawGivenTrainEnginePredOnAutoArgWinForComparison(i, 7)
    # instance.Draw3dData(instance.data.train_data.get_group(args.engine_choice).iloc[:, 1:].to_numpy(), 10, 0)
    instance.Draw3dData(instance.data.bare_train_data.get_group(args.engine_choice).iloc[:, 5:].to_numpy(), 20, 10)

if __name__ == '__main__':
    with torch.no_grad():
        main()
