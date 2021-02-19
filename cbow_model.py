import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn

from collections import defaultdict

class CBOWModel(pl.LightningModule):
    def __init__(self, n, v, h, lr=100):
        super().__init__()
        self.n = n
        self.vocab_size = v
        self.hidden_layer_size = h
        self.V = nn.Parameter(torch.rand((v, h), requires_grad=True).double())
        self.U = nn.Parameter(torch.rand((h, v), requires_grad=True).double())
        self.softmax = nn.Softmax(dim=1)

        self.lr = lr
        self.Loss = nn.BCELoss()

    def forward(self, x):
        B = x.shape[0]
        h = torch.mean(torch.reshape(x, [B, self.n - 1, self.vocab_size]) @ self.V, axis=1)
        yhat = self.softmax(h @ self.U)
        return yhat

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)

        loss = self.Loss(yhat, y)*10000
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)

        loss = self.Loss(yhat, y)*10000
        # pp = self.perplexity() # TODO: how to compute perplexity
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('val_pp', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # def seq_pp(self, seq):
    #     pass

    def configure_optimizers(self):
        print(self.lr, list(self.parameters()))
        return torch.optim.Adam(self.parameters(), lr=self.lr)