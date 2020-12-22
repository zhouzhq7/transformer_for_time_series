import copy

import numpy as np
import pytorch_lightning as pl

from torch.utils.data.dataloader import DataLoader

class TSDataModule(pl.LightningDataModule):

    def __init__(self, filename, seq_len=32, stride=1, shuffle=True, batch_size=64):
        super().__init__()
        self.filename = filename
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.stride = stride

    def load_data(self):
        import pandas as pd
        data = pd.read_csv('./data/stock_k_lines/600559_adjustflag_2.csv')['close'].to_numpy()
        train_sample = 2600
        return data[:2600], data[2600:]

    def generate_toy_data(self, seq_len=1024):
        import numpy as np
        time = np.arange(0, 400, 0.1)
        amplitude = np.sin(time) + np.sin(time * 0.05) + np.sin(time * 0.12) * np.random.normal(-0.2, 0.2, len(time))
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(-1, 1))
        amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
        return amplitude

    def format_raw_data(self, raw_data):
        n_sample = len(raw_data)
        all_samples = []
        for i in range(n_sample - self.stride - self.seq_len):
            input = raw_data[i  : i + self.seq_len]
            target = raw_data[i + self.stride : i +  self.stride + self.seq_len]
            all_samples.append((input, target))

        # for i in range(n_sample - self.seq_len):
        #     input = raw_data[i  : i + self.seq_len]
        #     target = copy.deepcopy(input)
        #     input[-self.stride:] = 0
        #     all_samples.append((input, target))

        return all_samples

    def split_data(self, formatted_data, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0):
        n_sample = len(formatted_data)
        n_train_data = int(n_sample * train_ratio)
        n_test_data = int(n_sample * val_ratio)

        train_samples = formatted_data[:n_train_data]
        val_samples = formatted_data[n_train_data:n_train_data + n_test_data]
        test_samples = formatted_data[n_train_data + n_test_data: ]

        return train_samples, val_samples, test_samples


    def setup(self, stage=None):
        train_data, val_data = self.load_data()

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(train_data.reshape(-1, 1))
        train_data = scaler.transform(train_data.reshape(-1, 1)).reshape(-1)
        val_data = scaler.transform(val_data.reshape(-1, 1)).reshape(-1)

        # convert time series data to trainable format
        self.train_data = self.format_raw_data(train_data)
        self.val_data = self.format_raw_data(val_data)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

