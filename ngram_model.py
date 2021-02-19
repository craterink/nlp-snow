import pytorch_lightning as pl
import numpy as np

from collections import defaultdict

class NgramModel(pl.LightningModule):
    def __init__(self, n, v):
        super().__init__()
        self.n = n
        self.vocab_size = v
        self.ngrams = {}
        self.TOTAL_KEY = '$TOTAL$'

    def forward(self, hists):
        NULL_PREDICT = ''
        preds = []
        for hist in hists:
            if not hist in self.ngrams:
                preds.append(NULL_PREDICT)

            # considering possible next word wn
            # return argmax_wn count(wn) / count(hist)
            count_hist = self.ngrams[self.TOTAL_KEY]
            count_wns = sorted([(k,v) for k, v in self.ngrams[hist] if k != self.TOTAL_KEY], key=lambda pair : pair[1], reverse=True)
            if count_wns:
                # best_word_prob = count_wns[0][1] / count_hist # Do not need this right now
                best_word = count_wns[0][0]
                preds.append(best_word)
            else:
                preds.append(NULL_PREDICT) # TODO: maybe implement random 
        return preds

    def training_step(self, batch, batch_idx):
        ngrams = list(zip(*batch))

        # compute loss of this batch (no gradient)
        parsed_ngrams = np.array([(' '.join(ngram[:-1]), ngram[-1]) for ngram in ngrams])
        # words_hat = self.forward(hists) # Not sure we need this
        loss = self.ngram_perplexity(parsed_ngrams)
        avg_loss = sum(loss) / len(loss)
        self.log('train_loss', avg_loss / self.vocab_size, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # update ngrams:
        for ngram in ngrams:
            hist = ' '.join(ngram[:-1])
            word = ngram[-1]

            if not hist in self.ngrams:
                self.ngrams[hist] = {self.TOTAL_KEY : 0}
            if not word in self.ngrams[hist]:
                self.ngrams[hist][word] = 0
            self.ngrams[hist][word] += 1
            self.ngrams[hist][self.TOTAL_KEY] += 1
        
        return loss

    def validation_step(self, batch, batch_idx):
        ngrams = list(zip(*batch))

        # compute loss of this batch (no gradient)
        parsed_ngrams = np.array([(' '.join(ngram[:-1]), ngram[-1]) for ngram in ngrams])
        # words_hat = self.forward(hists) # Not sure we need this
        loss = self.ngram_perplexity(parsed_ngrams)
        avg_loss = sum(loss) / len(loss)
        self.log('val_loss', avg_loss / self.vocab_size, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def ngram_prob(self, ngram):
        # Use Laplace Smoothing to compute ngram probability
        # prob(w|hist) = (count(w, hist) + 1) / (vocab size + number of times hist seen)
        hist, word = ngram[0], ngram[1]
        if hist in self.ngrams:
            if word in self.ngrams[hist]:
                return (self.ngrams[hist][word] + 1) / (self.vocab_size + self.ngrams[hist][self.TOTAL_KEY])
            else:
                return 1/(self.vocab_size + self.ngrams[hist][self.TOTAL_KEY])
        else:
            return 1/self.vocab_size

    def ngram_perplexity(self, ngrams):
        # returns the average perplexity of ngram pairs (hists, words)
        probs = [self.ngram_prob(ngram) for ngram in ngrams]
        ngram_perplexities = [(1/p) for p in probs]
        return ngram_perplexities

    def configure_optimizers(self):
        return None