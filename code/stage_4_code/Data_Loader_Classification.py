import os
import pickle
import re
from os.path import exists

import numpy as np
import torchtext
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


class Data_Loader():

    def __init__(self, max_len):
        self.max_len = max_len
    def preprocess_text(self, text):
        review = re.sub(r'https?://\S+|www\.\S+', '', text)  # links
        review = re.sub('<.*?>', '', review)  # removing html tags

        pattern = r'([.,!?-])'
        review = re.sub(pattern, r' \1 ', review)  # add whitespaces between punctuation
        review = re.sub(r'\s{2,}', ' ', review)  # remove double whitespaces

        review = re.sub(r'(.)\1+', r'\1\1', review)

        review = re.sub(r"https://\S+|www\.\S+", '', review)  # URL
        review = re.sub(r"[^a-zA-Z]", ' ', review)  # punctuation
        review = re.sub(r' +', ' ', review)  # remove whitespaces

        s_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        review = ' '.join(word.lower() for word in review.split() if word not in s_words)
        review = ' '.join([lemmatizer.lemmatize(w) for w in review.split(' ')])
        tokens = review.split()
        return tokens
    def get_data(self):
        test_path = '../../data/stage_4_data/text_classification/test'
        train_path = '../../data/stage_4_data/text_classification/train'
        X_test = []
        y_test = []

        X_train = []
        y_train = []
        # Put all the test/neg words into the X_test set
        for filename in os.listdir(test_path + "/neg"):
            with open(os.path.join(test_path + "/neg", filename), 'r', encoding='utf8') as file:
                text = file.read()
                X_test.append(self.preprocess_text(text))
                y_test.append(0)

        # Put all the test/pos words into the X_test set
        for filename in os.listdir(test_path + "/pos"):
            with open(os.path.join(test_path + "/pos", filename), 'r', encoding='utf8') as file:
                X_test.append(self.preprocess_text(file.read()))
                y_test.append(1)

        # Put all the train/neg words into the X_train set
        for filename in os.listdir(train_path + "/neg"):
            with open(os.path.join(train_path + "/neg", filename), 'r', encoding='utf8') as file:
                X_train.append(self.preprocess_text(file.read()))
                y_train.append(0)

        # Put all the train/pos words into the X_train set
        for filename in os.listdir(train_path + "/pos"):
            with open(os.path.join(train_path + "/pos", filename), 'r', encoding='utf8') as file:
                X_train.append(self.preprocess_text(file.read()))
                y_train.append(1)

        return X_test, y_test, X_train, y_train

    def pad(self, data):
        padded = np.ones((len(data), self.max_len), dtype=int)
        for i, review in enumerate(data):
            if len(review) != 0:
                padded[i, -len(review):] = review[:self.max_len]
        return padded

    def numericalize_data(self, review, vocab):
        ids = [vocab[token] for token in review]
        return ids

    def run(self):
        testandtrain = {}
        if exists('IMDB_clean.pickle'):
            print('Already cleaned')
            with open('IMDB_clean.pickle', 'rb') as handle:
                testandtrain = pickle.load(handle)
        else:
            print("Cleaning Data")
            X_test, y_test, X_train, y_train = self.get_data()

            special_tokens = ['<unk>', '<pad>']
            vocab = torchtext.vocab.build_vocab_from_iterator(X_train + X_test, min_freq=5, specials=special_tokens)
            unk_index = vocab['<unk>']
            vocab.set_default_index(unk_index)

            X_test = self.pad([self.numericalize_data(review, vocab) for review in X_test])
            X_train = self.pad([self.numericalize_data(review, vocab) for review in X_train])
            testandtrain = {'train': {'X': X_train, 'Y': y_train}, 'test': {'X': X_test, 'Y': y_test}, 'vocab': vocab}
            with open('IMDB_clean.pickle', 'wb') as handle:
                pickle.dump(testandtrain, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return testandtrain['train'], testandtrain['test'], testandtrain['vocab']