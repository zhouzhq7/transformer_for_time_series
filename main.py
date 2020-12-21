import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from model import TransformerModel
from data_module import TSDataModule

def inference():
    eval_model = TransformerModel.load_from_checkpoint('./lightning_logs/version_0/checkpoints/epoch=8-step=539.ckpt',
                                                       d_model=250, n_heads=10, n_layers=1)
    eval_model.freeze()
    n_steps = 1000

    test_data = pd.read_csv('./data/toy_data/test.csv').to_numpy()
    train_data = pd.read_csv('./data/toy_data/train.csv').to_numpy()

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_data)
    test_data = torch.tensor(scaler.transform(test_data).reshape(-1)).float()

    with torch.no_grad():
        for i in range(0, n_steps):
            # data = torch.cat((test_data[-99:], torch.tensor([0]).float()))
            data = test_data[-100:, ]
            output = eval_model(data.reshape(-1, 1).unsqueeze(-1))
            output = torch.flatten(output)
            test_data = torch.cat((test_data, output[-1:]))

    test_data = test_data.cpu().view(-1)

    # I used this plot to visualize if the model pics up any long therm struccture within the data.
    plt.plot(test_data[600:], color="red")
    plt.plot(test_data[600:1000], color="blue")
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.show()
    pass

def train():
    # data module
    dm = TSDataModule("", seq_len=100, batch_size=32)
    dm.setup()
    # model
    model = TransformerModel(250, 10, 1)

    # trainer
    trainer = pl.Trainer(gpus=[0], gradient_clip_val=0.7)

    trainer.fit(model=model, datamodule=dm)
    # prediction

    pass

if __name__=="__main__":
    inference()
    # train()