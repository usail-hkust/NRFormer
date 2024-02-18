import logging
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler

from src.utils import Scaler
from src import utils


class RadiationDataProcessing:
    def __init__(self, config):

        self.traffic_data = {}

        self.config = config

        self.dataset = config['dataset']
        self.data_path = config['data_path']
        self.num_sensors = config['num_sensors']

        self.train_prop = config['train_prop']
        self.valid_prop = config['valid_prop']

        self.in_length = config['in_length']
        self.out_length = config['out_length']
        self.in_channels = config['in_channels']
        self.batch_size = config['batch_size']
        self.adj_type = config['adj_type']

        self.nodeID = self.read_idx()

        # get several adjacency matrix
        self.adj_mx_01, self.adj_mx_dcrnn, self.adj_mx_gwn = self.read_adj_mat()

        self.dataloader = {}

        self.loc_feature = self.read_loc()
        self.dataloader['loc_feature'] = self.loc_feature

        # self.build_graph()
        self.build_data_loader()

    def build_graph(self):
        logging.info('initialize graph')

        for dim in range(self.adj_mats.shape[-1]):
            values = self.adj_mats[:, :, dim][self.adj_mats[:, :, dim] != np.inf].flatten()
            self.adj_mats[:, :, dim] = np.exp(-np.square(self.adj_mats[:, :, dim] / (values.std() + 1e-8)))

    def read_loc(self):
        loc = pd.read_csv(os.path.join(self.data_path, self.dataset, 'location_info.csv'), )
        loc_ft = np.zeros((self.num_sensors, 2))
        for i in range(loc.shape[0]):
            loc_ft[i, :] = loc.iloc[i, 2:4].tolist()
        normalized_loc_ft = (loc_ft - loc_ft.mean(axis=0)) / (loc_ft.std(axis=0) + 1e-8)
        return normalized_loc_ft

    def build_data_loader(self):

        train_traffic, valid_traffic, test_traffic = self.read_traffic()

        train_noaa, valid_noaa, test_noaa = {}, {}, {}
        if len(self.config['noaa_list']) > 0:
            for name in self.config['noaa_list']:
                # print(self.config['noaa_list'])
                train_noaa[name], valid_noaa[name], test_noaa[name] = self.read_noaa(tag=name)

        train_data = train_traffic[list(self.nodeID.keys())]
        self.scaler = Scaler(train_data.values, missing_value=0)

        # data for training & evaluation
        self.train = self.get_data_loader(train_traffic, train_noaa, shuffle=True, tag='train')
        self.valid = self.get_data_loader(valid_traffic, valid_noaa, shuffle=False, tag='valid')
        self.test = self.get_data_loader(test_traffic, test_noaa, shuffle=False, tag='test')

    def get_data_loader(self, data, noaa, shuffle, tag):

        if len(data) == 0:
            return 0
        num_timestamps = data.shape[0]

        data_time = data.iloc[:, 0]
        data_time = pd.to_datetime(data_time, utc=None)
        # data_time = data_time.dt.tz_localize(None)
        self.traffic_data[tag+'_data'] = data

        data = data[list(self.nodeID.keys())]

        # fill missing value
        data_fill = self.fill_traffic(data)

        # transform data distribution
        in_data = np.expand_dims(self.scaler.transform(data_fill.values), axis=-1)  # [T, N, 1]

        if len(self.config['noaa_list']) > 0:
            for name in self.config['noaa_list']:
                # 将DataFrame转换为ndarray
                array = noaa[name].values  # 或者 df.to_numpy()
                # 使用MinMaxScaler进行归一化
                # scaler = MinMaxScaler()
                # normalized_array = scaler.fit_transform(array)
                # d = np.expand_dims(normalized_array, axis=-1)
                # in_data = np.concatenate([in_data, d], axis=-1)  # [T, N, D]

                # 或者不使用sklearn，手动进行归一化
                # 注意：这里假设数据中没有零方差的特征（即每个特征的最大值和最小值不同）
                # min_vals = array.min(axis=0)
                # max_vals = array.max(axis=0)
                # normalized_array_manual = (array - min_vals) / (max_vals - min_vals)

                normalized_array = (array - array.mean(axis=0)) / (array.std(axis=0) + 1e-8)
                d = np.expand_dims(normalized_array, axis=-1)
                in_data = np.concatenate([in_data, d], axis=-1)  # [T, N, D]
        out_data = np.expand_dims(data.values, axis=-1)  # [T, N, 1]

        # create inputs & labels
        inputs, labels = [], []
        for i in range(self.in_length):
            temp = in_data[i: num_timestamps + 1 - self.in_length - self.out_length + i]
            inputs += [temp]
        for i in range(self.out_length):
            temp = out_data[self.in_length + i: num_timestamps + 1 - self.out_length + i]
            labels += [temp]
        # inputs = np.stack(inputs).transpose((1, 3, 2, 0))
        # labels = np.stack(labels).transpose((1, 3, 2, 0))
        inputs = np.stack(inputs).transpose((1, 0, 2, 3))
        labels = np.stack(labels).transpose((1, 0, 2, 3))

        # create dataset
        dataset = TensorDataset(
            torch.from_numpy(inputs).to(dtype=torch.float),
            torch.from_numpy(labels).to(dtype=torch.float)
        )

        # create sampler
        sampler = SequentialSampler(dataset)
        if shuffle:
            sampler = RandomSampler(dataset, replacement=True, num_samples=self.batch_size)
        else:
            sampler = SequentialSampler(dataset)

        # create dataloader
        data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, sampler=sampler,
                                 num_workers=4, drop_last=False)

        self.dataloader[tag+'_loader'] = DataLoaderM(inputs, labels, self.batch_size)
        self.dataloader['x_'+tag] = inputs
        self.dataloader['y_'+tag] = labels

        return data_loader

    def read_idx(self):

        with open(os.path.join(self.data_path, self.dataset, 'node_id.txt'), mode='r', encoding='utf-8') as f:
            ids = f.read().strip().split('\n')
        idx = {}
        for i, id in enumerate(ids):
            idx[id] = i
        return idx

    def read_adj_mat(self):
        # 更改 邻接矩阵只保留距离信息
        graph_csv = pd.read_csv(os.path.join(self.data_path, self.dataset, 'node_distance_{}.csv'.format(self.config['distance'])),
                                dtype={'from': 'str', 'to': 'str'})

        # matrix from DCRNN
        adj_distance_dcrnn = np.zeros((self.num_sensors, self.num_sensors))
        adj_distance_dcrnn[:] = np.inf  # 无穷大

        # g = np.zeros((self.num_sensors, self.num_sensors, 1))
        # g[:] = np.inf

        # 0, 1 adjacency matrix
        adj_mx_01 = np.zeros((self.num_sensors, self.num_sensors))
        for k in range(self.num_sensors):
            adj_mx_01[k, k] = 1

        # for k in range(self.num_sensors):
        #     g[k, k] = 0

        for row in graph_csv.values:
            if row[0] in self.nodeID and row[1] in self.nodeID:
                # g[self.nodeID[row[0]], self.nodeID[row[1]]] = row[2]  # distance

                # dcrnn matrix
                adj_distance_dcrnn[self.nodeID[row[0]], self.nodeID[row[1]]] = row[2]  # distance

                # 01 adjacency matrix
                adj_mx_01[self.nodeID[row[0]], self.nodeID[row[1]]] = 1  # 0, 1

        # gt = np.transpose(g, (1, 0, 2))
        # g_matrix = np.concatenate([g, gt], axis=-1)

        # dcrnn matrix
        distances = adj_distance_dcrnn[~np.isinf(adj_distance_dcrnn)].flatten()
        std = distances.std()
        adj_mx = np.exp(-np.square(adj_distance_dcrnn / std))
        # Make the adjacent matrix symmetric by taking the max.
        # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

        # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
        adj_mx[adj_mx < 0.1] = 0
        adj_mx_dcrnn = adj_mx

        # GraphWaveNet matrix
        adj_mx_gwn = None

        if self.adj_type == "scalap":
            adj = [utils.calculate_scaled_laplacian(adj_mx)]
        elif self.adj_type == "normlap":
            adj = [utils.calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
        elif self.adj_type == "symnadj":
            adj = [utils.sym_adj(adj_mx)]
        elif self.adj_type == "transition":
            adj = [utils.asym_adj(adj_mx)]
        elif self.adj_type == "doubletransition":
            # GraphWaveNet matrix
            adj_mx_gwn = [utils.asym_adj(adj_mx), utils.asym_adj(np.transpose(adj_mx))]
        elif self.adj_type == "identity":
            adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
        else:
            error = 0
            assert error, "adj type not defined"

        return adj_mx_01, adj_mx_dcrnn, adj_mx_gwn


    def read_noaa(self, tag):
        # data = pd.read_hdf(os.path.join(data_path, self._path, 'traffic.h5'))
        data = pd.read_csv(os.path.join(self.data_path, self.dataset, 'noaa', '{}.csv'.format(tag)))
        # data.iloc[:, 1:] = data.iloc[:, 1:] * 1000
        # self.data_time = data.iloc[:, 0]

        num_train = int(data.shape[0] * self.train_prop)
        num_valid = int(data.shape[0] * self.valid_prop)
        num_test = data.shape[0] - num_train - num_valid

        train = data[:num_train].copy()
        valid = data[num_train: num_train + num_valid].copy()
        test = data[-num_test:].copy()

        return train, valid, test


    def read_traffic(self):
        # data = pd.read_hdf(os.path.join(data_path, self._path, 'traffic.h5'))
        data = pd.read_csv(os.path.join(self.data_path, self.dataset, 'data.csv'))
        # data.iloc[:, 1:] = data.iloc[:, 1:] * 1000
        self.data_time = data.iloc[:, 0]

        self.num_train = int(data.shape[0] * self.train_prop)
        self.num_valid = int(data.shape[0] * self.valid_prop)
        self.num_test = data.shape[0] - self.num_train - self.num_valid

        train = data[:self.num_train].copy()
        valid = data[self.num_train: self.num_train + self.num_valid].copy()
        test = data[-self.num_test:].copy()

        return train, valid, test

    def fill_traffic(self, data):
        # data = data[list(self.nodeID.keys())]
        data = data.copy()
        data[data < 1e-5] = float('nan')
        data = data.fillna(method='pad')
        data = data.fillna(method='bfill')
        return data


class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()