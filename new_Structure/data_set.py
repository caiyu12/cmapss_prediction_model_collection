from bisect import bisect_right
# REMIND: bisect_left will fine a number's position
#  in oriented list, treating the two neighbour numbers
#  a left-closed, right-open interval and return the
#  interval's N.O + 1.

import torch
import numpy as np
from torch.utils.data import Dataset
from pandas.core.groupby.generic import DataFrameGroupBy
from torch import Tensor

# REMIND: the class is a subclass of torch.utils.data.Dataset,
#  which is a abstract class for representing a dataset.
#  The subclass should implement two methods: __len__ and
#  __getitem__. The former should return the size of the
#  dataset, and the latter should return the i-th sample
#  of the dataset.
class TrainDataset(Dataset):
    def __init__(
            self,
            data_group : DataFrameGroupBy,
            targ_group : DataFrameGroupBy,
            window_size: int,

    ) -> None:
        assert data_group.ngroups == targ_group.ngroups, 'Training data and target group must have the same number of groups'

        self.num_total = data_group.ngroups
        self.data_group = data_group
        self.targ_group = targ_group
        self.window_size = window_size

        __window_num_total = 0
        __window_index_list = [] # REMIND:
        #   the list is used to store the start window's index of each unit's data
        #   eg: ..list = [0, 156, 312, 468, ...] means:
        #     the first unit's data starts from index 0,
        #     the second unit's data starts from index 156,
        #     the third unit's data starts from index 312...
        #  and then we can use bisect_right to find the exact unit that the index belongs to.
        __window_index_list.append(__window_num_total) # appending the first unit's start index, 0

        __padding_list = []
        # the following loop is used to calculate the start index of each unit's data
        for i in range(1, self.num_total + 1):
            if self.data_group.get_group(i).shape[0] >= self.window_size:
                # the core calculation of indexing
                __window_num_indiv = (self.data_group.get_group(i).shape[0] - self.window_size) + 1
                # calculate the total number of windows
                __window_num_total += __window_num_indiv
            else:
                __padding_list.append(i)
                __window_num_indiv = 1
                __window_num_total += __window_num_indiv

            __window_index_list.append(__window_num_total)

        self.data_length = __window_num_total
        self.index_list = __window_index_list

        # REMIND: following aimed at higher performance
        self.data_list = list()
        self.targ_list = list()

        for i in range(1, self.num_total + 1):
            __data_engine_origin = self.data_group.get_group(i).to_numpy()
            if i in __padding_list: # REMIND:padding using the first row
                __data_missing_row = self.window_size - __data_engine_origin.shape[0]

                __data_engine_indiv = torch.zeros(size=(self.window_size, __data_engine_origin.shape[1]), dtype=torch.float32)
                __data_engine_indiv[:__data_missing_row, :] = torch.tensor(data=__data_engine_origin[0, :], dtype=torch.float32)
                __data_engine_indiv[__data_missing_row:, :] = torch.tensor(data=__data_engine_origin[:, :], dtype=torch.float32)

                self.data_list.append(__data_engine_indiv)
            else:
                __data_engine_indiv = torch.tensor(data=__data_engine_origin, dtype=torch.float32)
                self.data_list.append(__data_engine_indiv)

        for i in range(1, self.num_total + 1):
            __targ_engine_origin = self.targ_group.get_group(i).to_numpy()

            if i in __padding_list: # REMIND:padding using the first row
                __targ_missing_row   = self.window_size - __targ_engine_origin.shape[0]

                __targ_engine_indiv = torch.zeros(size=(self.window_size, __targ_engine_origin.shape[1]), dtype=torch.float32)
                __targ_engine_indiv[:__targ_missing_row, :] = torch.tensor(data=__targ_engine_origin[0, :], dtype=torch.float32)
                __targ_engine_indiv[__targ_missing_row:, :] = torch.tensor(data=__targ_engine_origin[:, :], dtype=torch.float32)

                self.targ_list.append(__targ_engine_indiv)
                # print('TRAIN: Engine {} used padding: {} -> {}'.format(i, __targ_engine_origin.shape[0], self.window_size))
            else:
                __targ_engine_indiv = torch.tensor(data=__targ_engine_origin, dtype=torch.float32)
                self.targ_list.append(__targ_engine_indiv)

    def __len__(self) -> int:
        # print("--train-__len__- %s seconds ---" % (time.time() - start_time))
        return self.data_length

    def __getitem__(self, index : int) -> (Tensor, Tensor):
        position = bisect_right(self.index_list, index)
        # # REMIND: position sequence starting from 1
        index_shifted = index - self.index_list[position - 1]
        index_slided  = index_shifted

        data_tensor  = self.data_list[position-1][index_slided:index_slided+self.window_size, 1:]
        target_tensor = self.targ_list[position-1][index_slided+self.window_size-1, 1:]

        return data_tensor, target_tensor

class TestDataset(Dataset):
    def __init__(
            self,
            data_group : DataFrameGroupBy,
            targ_group : DataFrameGroupBy,
            accept_window : int,
    ) -> None:
        assert data_group.ngroups == targ_group.ngroups, 'Testing data and target group must have the same number of groups'

        self.num_total  = data_group.ngroups
        self.data_group = data_group
        self.targ_group = targ_group
        self.accept_window = accept_window

        self.data_list = list()
        self.targ_list = list()
        for i in range(1, self.num_total + 1):
            __data_engine_origin = self.data_group.get_group(i).to_numpy()

            if accept_window > __data_engine_origin.shape[0]: # REMIND:padding using the first row
                __data_missing_row = self.accept_window - __data_engine_origin.shape[0]

                __data_engine_indiv = torch.zeros(size=(self.accept_window, __data_engine_origin.shape[1]), dtype=torch.float32)
                __data_engine_indiv[:__data_missing_row, :] = torch.tensor(data=__data_engine_origin[0, :], dtype=torch.float32)
                __data_engine_indiv[__data_missing_row:, :] = torch.tensor(data=__data_engine_origin[:, :], dtype=torch.float32)

                self.data_list.append(__data_engine_indiv)
                # print('TEST: Engine {} used padding: {} -> {}'.format(i, __data_engine_origin.shape[0], self.accept_window))
            else:
                __data_engine_indiv = torch.tensor(data=__data_engine_origin, dtype=torch.float32)
                self.data_list.append(__data_engine_indiv)

        for i in range(1, self.num_total + 1):
            __targ_engine_origin = self.targ_group.get_group(i).to_numpy()

            self.targ_list.append(__targ_engine_origin)

    def __len__(self):
        return self.num_total

    def __getitem__(self, index):
        test_tensor = self.data_list[index][:, 1:]

        target_tensor = self.targ_list[index][0, 1:]

        return test_tensor, target_tensor
