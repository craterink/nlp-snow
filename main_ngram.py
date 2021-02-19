from fire import Fire
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb

from multiprocessing import Pool
from itertools import repeat

from wikitext_data import *
from ngram_model import *

def main(train_file, valid_file, test_file, n_vals, batch_size):
    for n in [int(val) for val in n_vals] if not type(n_vals) == int else [n_vals]:
        run = wandb.init(project="nlp-snow", reinit=True)
        run.name = 'ngram_n={}'.format(n)
        wandb_logger = WandbLogger(name='ngram_n={}'.format(n),project='nlp-snow')

        dl_kwargs = {
            'batch_size' : batch_size,
            'num_workers' : 4
        }
        datasets = list(map(WikitextDataset, [train_file, valid_file, test_file], [simple_tokenizer] * 3, [n] * 3))
        dataloaders = [DataLoader(ds, shuffle=True if idx == 0 else False, **dl_kwargs) for idx, ds in enumerate(datasets)]
        vocab_size = len(set([token for d in datasets for token in d.tokens]))
        print('Vocab Size = ' + str(vocab_size))
        model = NgramModel(n, vocab_size)
        trainer = pl.Trainer(automatic_optimization=False, logger=wandb_logger, callbacks=[EarlyStopping(monitor='val_loss')])

        trainer.fit(model, *dataloaders[:2])

if __name__ == "__main__":
    Fire(main)