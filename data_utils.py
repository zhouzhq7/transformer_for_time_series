import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_inout_sequences(input_data, tw, out_win_size):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + out_win_size: i + tw + out_win_size]
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)

def get_toy_data(in_win_size, device):
    time        = np.arange(0, 400, 0.1)    
    amplitude   = np.sin(time) + np.sin(time * 0.05) +np.sin(time * 0.12) * np.random.normal(-0.2, 0.2, len(time))
    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
    
    sampels = 2600
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:]

    train_sequence = create_inout_sequences(train_data, in_win_size)
    test_data = create_inout_sequences(test_data, in_win_size)

    return train_sequence.to(device), test_data.to(device)

def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]    
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    return input, target