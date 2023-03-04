'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pickle
import os
import re
import string
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from os.path import exists
import numpy as np


class Dataset_Loader(dataset):
    test_data = None
    test_path = None
    test_file_name = None

    train_data = None
    train_path = None
    train_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)


    def preprocess_text(self, text):
        text = text.lower()
        text_numfree = re.sub(r'\d+', '', text)
        text_punctfree = text_numfree.translate(str.maketrans('', '', string.punctuation))

        # coming back to removing stop words
        text_embd_tokens = word_tokenize(text_punctfree)
        tok_clean_sw = [word for word in text_embd_tokens if not word in stopwords.words()]
        text_tok_clean = (" ").join(tok_clean_sw)

        text_stripped = re.sub('\s+', ' ', text_tok_clean).strip()

        return text_stripped

        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # str in square brackets
        text = re.sub('(%s)' % re.escape(string.punctuation), '', text)  # str in round brackets
        joke = re.sub(r'https?://\S+|www\.\S+', '', text)  # links

        pattern = r'([.,!?-])'
        joke = re.sub(pattern, r' \1 ', joke)  # add whitespaces between punctuation
        joke = re.sub(r'\s{2,}', ' ', joke)  # remove double whitespaces

        joke = re.sub(r'(.)\1+', r'\1\1', joke)
        joke = re.sub('\w*\d\w*', '', joke)

        joke = re.sub(r"[^a-zA-Z]", ' ', joke)  # punctuation
        joke = re.sub(r' +', ' ', joke)  # remove whitespaces

        joke = joke.lower()
        joke = joke.split()
        return joke

    def data_split(self, test_size=0.2):


        #x_t = torch.Tensor()
        #y_t = torch.Tensor()

        #vals = x.shape[0]

        with open('', r) as file:
            text = file.readlines()

        x = []
        y = []
        for line in text:
            proc_text = self.preprocess(line)
            vals = proc_text.split()
            x.append(vals[:-1])
            y.append(vals[:-1])

        sp_point = int(x[0] * (1 - test_size))

        x_t = torch.Tensor(x)
        y_t = torch.Tensor(y)





        x_train = x_t[:sp_point]
        y_train = y_t[:sp_point]
        x_test = x_t[sp_point:]
        y_test = y_t[sp_point:]

        return x_train, y_train, x_test, y_test





    def load(self):
        print('loading data...')
        text_gen_path = '../../data/stage_4_data/text_generation/data'
        X_test = []
        y_test = []

        X_train = []
        y_train = []

        for filename in os.listdir(text_gen_path + "/data"):
            with open(os.path.join(text_gen_path + "/data", filename), 'r', encoding='utf8') as file:
                text = file.read()
                X_test.append(self.preprocess_text(text))

        #X_train, y_train, X_test, y_test = self.data_split()

        return {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}