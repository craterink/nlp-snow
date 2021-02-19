from fire import Fire
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb

from multiprocessing import Pool
from itertools import repeat

from wikitext_data import *
from cbow_model import *

def main(train_file, valid_file, test_file, n, batch_size, h=12, lr=1000):
    run = wandb.init(project="nlp-snow", reinit=True)
    run_name = 'cbow_bs={batch_size}'
    run.name = run_name
    wandb_logger = WandbLogger(name=run_name,project='nlp-snow')

    dl_kwargs = {
        'batch_size' : batch_size,
        'num_workers' : 4
    }
    predetermined_vocab_size, vocab_lookup = vocab_lookup_info([train_file, valid_file, test_file], simple_tokenizer)
    datasets = list(map(WikitextDatasetOneHot, [train_file, valid_file, test_file], [simple_tokenizer] * 3, [predetermined_vocab_size] * 3, [vocab_lookup] * 3, [n] * 3))
    dataloaders = [DataLoader(ds, shuffle=True if idx == 0 else False, **dl_kwargs) for idx, ds in enumerate(datasets)]
    model = CBOWModel(n, predetermined_vocab_size, h, lr)
    trainer = pl.Trainer(automatic_optimization=False, logger=wandb_logger, callbacks=[EarlyStopping(monitor='val_loss')])
    
    print('Vocab Size = ' + str(predetermined_vocab_size))
    print('Train Corpus Size = ' + str(len(datasets[0].tokens)) )
    
    trainer.fit(model, *dataloaders[:2])

if __name__ == "__main__":
    Fire(main)