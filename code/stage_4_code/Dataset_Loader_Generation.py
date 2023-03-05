import pickle
import re
from os.path import exists
import string
import torchtext
import pandas as pd
from nltk import tokenize
import pickle
import re
import string
from os.path import exists

import pandas as pd
import torchtext
from nltk import tokenize


class Data_Loader():

    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
    def preprocess_text(self, text):
        text = text.lower()
        text = tokenize.sent_tokenize(text)
        text = " <EOS> ".join(text)

        text = re.sub(r"(?<!<EOS>)[%s]" % re.escape(string.punctuation), '', text)
        text = re.sub(r"(?<!<EOS>)(%s)" % re.escape(string.punctuation), '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # links

        pattern = r'([.,!?-])'
        text = re.sub(pattern, r' \1 ', text)  # add whitespaces between punctuation
        text = re.sub(r'\s{2,}', ' ', text)  # remove double whitespaces

        text = re.sub(r'(.)\1+', r'\1\1', text)
        text = re.sub('\w*\d\w*', '', text)

        text = re.sub(r"(?!<EOS>)[^a-zA-Z]", ' ', text)  # punctuation
        text = re.sub(r' +', ' ', text)  # remove whitespaces

        text = text.replace('EOS', '<EOS>')
        text = text.split()
        if not text[-1].endswith('<EOS>'):
            text.append('<EOS>')
        text = [word for i, word in enumerate(text) if i == 0 or word != '<EOS>' or text[i-1] != '<EOS>']
        text.append('<EOJ>')
        return text

    def get_data(self):
        path = '../../data/stage_4_data/text_generation/data'
        df = pd.read_csv(path)
        # Remove Newlines
        df["Joke"] = df["Joke"].replace(r'\n', ' ', regex=True)
        df["Joke"] = df["Joke"].replace(r'\r', ' ', regex=True)
        jokes = df['Joke'].values.tolist()
        jokes = [self.preprocess_text(joke) for joke in jokes]
        return jokes

    def numericalize_data(self, review, vocab):
        ids = [vocab[token] for token in review]
        return ids

    def run(self):
        testandtrain = {}
        if exists('jokes_clean.pickle'):
            print('Already cleaned')
            with open('jokes_clean.pickle', 'rb') as handle:
                testandtrain = pickle.load(handle)
        else:
            print("Cleaning Data")
            X_train = self.get_data()

            special_tokens = ['<unk>', '<pad>','<EOS>', '<EOJ>']
            vocab = torchtext.vocab.build_vocab_from_iterator(X_train, min_freq=1, specials=special_tokens)
            unk_index = vocab['<unk>']
            vocab.set_default_index(unk_index)

            X_test = [self.numericalize_data(review, vocab) for review in X_train]

            testandtrain = {'X': X_test, 'vocab': vocab}
            with open('jokes_clean.pickle', 'wb') as handle:
                pickle.dump(testandtrain, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return testandtrain['X'], testandtrain['vocab']