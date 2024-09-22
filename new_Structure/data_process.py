from data_set import TrainDataset, TestDataset

from argparse import Namespace
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from numpy import ndarray
from pandas.core.groupby.generic import DataFrameGroupBy
import os
import pandas as pd
import torch

class CMAPSS_Data_Process():
    def __init__(self, arg: Namespace) -> None:
        self.arg = arg
        self.loadData(
            root=self.arg.directory,
            dataset=self.arg.dataset,
            max_rul=self.arg.max_rul
        )

    def loadData(
            self,
            root : str,
            dataset   : str,
            max_rul   : int,
    ) -> None:
        train_data_pt = os.path.join(root, 'CMAPSSData',  'train_'+ dataset +'.txt')
        assert os.path.exists(train_data_pt), 'data path does not exist: {:}'.format(train_data_pt)

        test_data_pt = os.path.join(root, 'CMAPSSData', 'test_'+ dataset +'.txt')
        assert os.path.exists(test_data_pt), 'data path does not exist: {:}'.format(test_data_pt)

        test_truth_pt = os.path.join(root, 'CMAPSSData', 'RUL_'+ dataset +'.txt')
        assert os.path.exists(test_truth_pt), 'data path does not exist: {:}'.format(test_truth_pt)

        column_names = [
            'id', 'cycle',
            'setting1', 'setting2', 'setting3',
            's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
            's9', 's10', 's11', 's12', 's13', 's14', 's15',
            's16', 's17', 's18', 's19', 's20', 's21'
        ]
        train_data_df = pd.read_csv(train_data_pt, sep=r'\s+', header=None, names=column_names)
        test_data_df  = pd.read_csv(test_data_pt, sep=r'\s+', header=None, names=column_names)
        test_target_df = pd.read_csv(test_truth_pt, sep=r'\s+', header=None, names=['RUL'])
        self.train_engine_num = train_data_df['id'].nunique()
        self.test_engine_num  = test_data_df['id'].nunique()

        (self.train_data, self.train_target,
         self.test_data, self.test_target) = self.dataProcess(train_data_df, test_data_df, test_target_df, max_rul)

    def getTrainDataloader(
            self,
            window_size : int,
            batch_size  : int,
            memory_pinned: bool
    ) -> DataLoader:

        train_dataset = TrainDataset(
        data_group=self.train_data,
        targ_group=self.train_target,
        window_size=window_size
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            pin_memory=memory_pinned
        )

        return train_dataloader

    def getTestDataloader(self, batch_size: int, memory_pinned: bool) -> DataLoader:
        # REMIND: Here batch_size must be 1
        test_dataset = TestDataset(
            data_group=self.test_data,
            targ_group=self.test_target,
            accept_window=self.arg.accept_window,
            max_window=self.arg.window_size_tuple[-1]
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=memory_pinned
        )

        return test_dataloader

    def sig_testdata(self,engine_id):
        test_dataset = TestDataset(
            data_group=self.test_data,
            targ_group=self.test_target,
            accept_window=self.arg.accept_window,
            max_window=self.arg.window_size_tuple[-1]
        )
        test_data, target_data = test_dataset.__getitem__(engine_id)
        return test_data, target_data


    def dataProcess(
            self,
            train_data : pd.DataFrame,
            test_data  : pd.DataFrame,
            test_truth : pd.DataFrame,
            max_rul    : int
    ) -> (DataFrameGroupBy, DataFrameGroupBy, DataFrameGroupBy, DataFrameGroupBy):
        # Processing train data
        tmp_train_RUL = pd.DataFrame(train_data.groupby('id')['cycle'].max()).reset_index()
        tmp_train_RUL.columns = ['id', 'max']
        train_data = train_data.merge(tmp_train_RUL, on=['id'], how='left')
        train_targ = pd.DataFrame(data=train_data['id']).join(pd.DataFrame(data=(train_data['max'] - train_data['cycle']), columns=['target']))
        train_targ['target'] = train_targ['target'].clip(upper=125)

        train_data.drop('max', axis=1, inplace=True)
        train_data.drop(['s1', 's5', 's6', 's10', 's16', 's18', 's19'], axis =1, inplace = True)

        train_data['setting1'] = train_data['setting1'].round(1)

        # Processing test data
        test_targ = pd.DataFrame(data=test_truth).reset_index()
        test_targ = test_targ.rename(columns={'index' : 'id', 'RUL' : 'target'})
        test_targ['id'] = test_targ['id'] + 1
        test_targ['target'] = test_targ['target'].clip(upper=125)

        test_data.drop(['s1', 's5', 's6', 's10', 's16', 's18', 's19'], axis =1, inplace = True)

        test_data['setting1'] = test_data['setting1'].round(1)

        # Normalizing data
        # Remind: fault mode not concerned
        bare_train_data = train_data.iloc[:, 2:]
        bare_test_data = test_data.iloc[:, 2:]
        scaler = MinMaxScaler()

        grouped_train = bare_train_data.groupby('setting1')
        grouped_test = bare_test_data.groupby('setting1')
        train_normalized = pd.DataFrame(columns = bare_train_data.columns[3:])
        test_normalized = pd.DataFrame(columns = bare_test_data.columns[3:])
        for train_idx, train in grouped_train:

            scaled_train : ndarray = scaler.fit_transform(train.iloc[:, 3:])
            scaled_train_combine = pd.DataFrame(
                data=scaled_train,
                index=train.index,
                columns=bare_train_data.columns[3:])
            train_normalized = pd.concat([train_normalized, scaled_train_combine]) if not train_normalized.empty \
                else scaled_train_combine

            for test_idx, test in grouped_test:
                if train_idx == test_idx:
                    scaled_test : ndarray = scaler.transform(test.iloc[:, 3:])
                    scaled_test_combine   = pd.DataFrame(
                        data=scaled_test,
                        index=test.index,
                        columns=bare_test_data.columns[3:])
                    test_normalized = pd.concat([test_normalized, scaled_test_combine]) if not test_normalized.empty \
                        else scaled_test_combine

        bare_train_normalized_df = train_normalized.sort_index()
        bare_test_normalized_df = test_normalized.sort_index()
        # bare_train_normalized_df = pd.DataFrame(data=[train_data['id'], bare_train_normalized_df])
        # bare_test_normalized_df = pd.DataFrame(data=[test_data['id'], bare_test_normalized_df])
        bare_train_normalized_df = pd.DataFrame(data=train_data['id']).join(bare_train_normalized_df)
        bare_test_normalized_df  = pd.DataFrame(data=test_data['id']).join(bare_test_normalized_df)

        train_settings = train_data.iloc[:, 1:5]
        test_settings  = test_data.iloc[:, 1:5]
        train_settings_normalized : ndarray = scaler.fit_transform(train_settings)
        test_settings_normalized  : ndarray = scaler.transform(test_settings)
        train_settings_normalized_df = pd.DataFrame(
            data=train_settings_normalized,
            index=train_data.index,
            columns=train_data.columns[1:5]
        )
        test_settings_normalized_df  = pd.DataFrame(
            data=test_settings_normalized,
            index=test_data.index,
            columns=test_data.columns[1:5]
        )

        train_settings_normalized_df = pd.DataFrame(data=train_data['id']).join(train_settings_normalized_df)
        test_settings_normalized_df  = pd.DataFrame(data=test_data['id']).join(test_settings_normalized_df)

        train_targ['target'] = train_targ['target'].apply(lambda x: (x/max_rul))
        test_targ['target']  = test_targ['target'].apply(lambda x: (x/max_rul))

        return (bare_train_normalized_df.groupby(by='id'), train_targ.groupby(by='id'),
                bare_test_normalized_df.groupby(by='id'), test_targ.groupby(by='id'))

