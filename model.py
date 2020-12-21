import torch
import numpy as np
import pytorch_lightning as pl

from torch import nn
from pos_encoder import PositionalEncoding


class TransformerModel(pl.LightningModule):

    def __init__(self, d_model, n_heads, n_layers, lr=1e-3, dropout=0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.lr = lr
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.Linear(d_model, 1)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _generate_mask(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        self._generate_mask(src=src, has_mask=has_mask)        
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)

        return output
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        x = x.transpose(0, 1).unsqueeze(-1)
        y = y.transpose(0, 1).unsqueeze(-1)
        output = self(x)
        loss = nn.MSELoss()(output, y)
        self.log('val_loss', loss)


    def training_step(self, batch, n_batch, has_mask=True):
        x, y = batch
        # print(x.shape)
        x = x.float()
        y = y.float()
        x = x.transpose(0, 1).unsqueeze(-1)
        y = y.transpose(0, 1).unsqueeze(-1)
        if has_mask:
            device = x.device
            if self.src_mask is None or self.src_mask.size(0) != len(x):
                mask = self._generate_square_subsequent_mask(len(x)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        pos_encoded_x = self.pos_encoder(x)
        output = self.transformer_encoder(pos_encoded_x, self.src_mask)
        output = self.decoder(output)

        loss = nn.MSELoss()(output, y)

        self.log('train_loss', loss)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.95
        )

        return [optimizer], [scheduler]

# train_data_x = np.array(np.random.uniform(-1, 1, (128, 100, 1))).astype(float)
# train_data_y = np.array(np.random.uniform(-1, 1, (128, 100, 1))).astype(float)

# val_data_x = np.array(np.random.uniform(-1, 1, (16, 100, 1)))
# val_data_y = np.array(np.random.uniform(-1, 1, (16, 100, 1)))
# train_data = [(train_data_x[i, :, :], train_data_y[i, :, :]) for i in range(len(train_data_x))]
# val_data = [(val_data_x[i, :, :], val_data_y[i, :, :]) for i in range(len(val_data_x))]

# model = TransformerModel(d_model=256, n_heads=8, n_layers=4)

# trainer = pl.Trainer()

# trainer.fit(model, train_dataloader=torch.utils.data.DataLoader(train_data),
# val_dataloaders=torch.utils.data.DataLoader(val_data))