from torch.utils.data import Dataset
import numpy as np

import json

def simple_tokenizer(string):
    """
    Removes non-alpha characters and lowercases. Converts stops to Splits by spaces.
    """
    keep_punc = '.!'
    denoised = ''
    for c in string:
        if c.isalpha():
            denoised += c
        elif c in keep_punc:
            denoised += ' ' + c + ' ' # make sure keep_punc is treated as separate token
        elif c == ' ':
            denoised += c
        elif c in '\n\r':
            denoised += ' '
    return denoised.split()

def vocab_lookup_info(files, tokenizer):
    # check for cache
    try:
        with open('raw/vocab_lookup.cache', 'r') as f:
            print('loading cache!!')
            vocab_lookup = json.load(f)
            return len(vocab_lookup.keys()), vocab_lookup
    except Exception as ex:
        pass

    total_tokens = []
    for f in files:
         with open(f, 'r') as fi:
            total_tokens += tokenizer(fi.read())
    tokens_unique = set(total_tokens)
    vocab_lookup = {k:i for i, k in enumerate(list(tokens_unique))} # induces some randomness

    try:
        with open('raw/vocab_lookup.cache', 'w') as f:
            json.dump(vocab_lookup, f)
    finally:
        pass

    return len(tokens_unique), vocab_lookup

class WikitextDataset(Dataset):
    def __init__(self, input_text_filepath, tokenizer, n=2):
        with open(input_text_filepath, 'r') as f:
            self.raw_text = f.read()
        self.tokens = tokenizer(self.raw_text)
        self.n = n

    def __len__(self):
        return len(self.tokens) - self.n + 1

    def __getitem__(self, idx):
        return self.tokens[idx:idx+self.n]

class WikitextDatasetOneHot(Dataset):
    def __init__(self, input_text_filepath, tokenizer, known_total_vocab_size, vocab_lookup, n=5):
        # n is the size of the context + middle word. index of middle word = n // 2 
        with open(input_text_filepath, 'r') as f:
            self.raw_text = f.read()
        self.tokens = tokenizer(self.raw_text)
        self.n = n
        self.middle_idx = n // 2
        self.vocab_size = known_total_vocab_size
        self.vocab_lookup = vocab_lookup

    def __len__(self):
        return len(self.tokens) - self.n + 1

    def __getitem__(self, idx):
        words = self.tokens[idx:idx + self.n]
        word_idxs = [self.vocab_lookup[w] for w in words] # 
        word_onehots = np.zeros((self.vocab_size, self.n))
        word_onehots[word_idxs, np.arange(self.n)] = 1
        context = np.concatenate([word_onehots[:, :self.middle_idx], word_onehots[:, self.middle_idx+1:]], axis=1)
        target = word_onehots[:, self.middle_idx]
        return context, target