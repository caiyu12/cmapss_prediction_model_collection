from test_model import *
from data_process import CMAPSS_Data_Process

from argparse import Namespace
import torch

class Train():
    def __init__(self, arg : Namespace, model) -> None:
        self.arg = arg
        data = CMAPSS_Data_Process(self.arg)



def args_config(dataset_choice : int) -> Namespace:
    arguments = Namespace(
        directory = '.\\',
        dataset   = 'FD00{}'.format(dataset_choice),
        epoch     = 41,
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_rul   = 125,

        memory_pinned = True,
        # REMIND: place model hyperparameters here
    )
    match dataset_choice:
        case 1:
            arguments.windows_size = 50
            arguments.batch_size   = 100
        case 2:
            arguments.windows_size = 50
            arguments.batch_size   = 100

        case 3:
            arguments.windows_size = 30
            arguments.batch_size   = 100

        case 4:
            arguments.windows_size = 50
            arguments.batch_size   = 100

        case _:
            raise ValueError("Invalid dataset choice")

    return arguments

def main() -> None:
    args = args_config(
        dataset_choice=2,
    )
    model = TSMixer(sensors=14, e_layers=4, d_model=36, seq_len=args.windows_size, pred_len=1, dropout=0.2)
    train = Train(args, model)


if __name__ == '__main__':
    main()
