from PyDictionary import PyDictionary
from snippets.pearson_dict import pearson_meaning
import pickle
import pandas as pd
import numpy as np
import random
import sys
sys.path.append('../../')
from util import *
from keras.preprocessing.sequence import pad_sequences

EMBEDDINGS_PATH = '../../nns/datasets/glove/glove.6B.300d.pickle'
EMBEDDINGS_PATH = '../../datasets/glove/embeddings_300.pickle'
LOCAL_WORDS = '/usr/share/dict/words'
WIKI_WORDS = 'wiki-100k.notatxt'

class DictTask:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.dict1 = PyDictionary().meaning
        self.dict2 = pearson_meaning
        # with open(EMBEDDINGS_PATH, 'rb') as handle:
            # self.weights = pickle.load(handle)
        # self.words_list = [x.strip().lower() for x in open(LOCAL_WORDS, 'r')]

    def load_from_file(self, file):
        with open(file, 'rb') as handle:
            dataset = pickle.load(handle)
        # random.shuffle(dataset)
        new_dataset, vocab, word_to_i = [], set([]), {}

        for m1, m2, w in dataset:
            # Get specific defs
            '''
            first_key = list(m1.keys())[0]
            m1 = m1[first_key][0]

            m2 = m2['senses'][0]['definition']
            m2 = m2 if type(m2) == str else m2[0]
            '''
            cleaned_m1, cleaned_m2 = clean_text(m1), clean_text(m2)

            # Tokenize
            new_m1, new_m2 = [], []

            for word in cleaned_m1:
                if word not in word_to_i.keys():
                    word_to_i[word] = len(word_to_i) + 1
                new_m1.append(word_to_i[word])

            for word in cleaned_m2:
                if word not in word_to_i.keys():
                    word_to_i[word] = len(word_to_i) + 1
                new_m2.append(word_to_i[word])

            if w not in word_to_i.keys():
                word_to_i[w] = len(word_to_i) + 1
            new_w = word_to_i[w]

            new_dataset.append((new_m1, new_m2, new_w, cleaned_m1, cleaned_m2, w))

        batches, i = [], 0

        while i + self.batch_size < len(new_dataset):
            batch = [[], [], [], [], [], []]

            for j in range(self.batch_size):
                new_m1, new_m2, new_w, m1, m2, w = new_dataset[i + j]
                batch[0].append(new_m1)
                batch[1].append(new_m2)
                batch[2].append(new_w)
                batch[3].append(m1)
                batch[4].append(m2)
                batch[5].append(w)

            i += j
            new_batch = [[], [], [], []]
            max_length1 = np.max([np.shape(v)[0] for v in batch[0]])
            max_length2 = np.max([np.shape(v)[0] for v in batch[1]])
            max_length = np.maximum(max_length1, max_length2)
            new_batch[0] = pad_sequences(batch[0], max_length, padding='post')
            new_batch[1] = pad_sequences(batch[1], max_length, padding='post')
            new_batch[2] = batch[2]
            new_batch[3] = batch[3:6]

            batches.append(new_batch)

        return batches[:-1], len(word_to_i), word_to_i

    def embed(self, word):
        if word not in self.weights.keys():
            dims = len(self.weights['a'])
            self.weights[word] = np.random.randn(dims, stddev=.7)
        return self.weights[word]

    def embed_sentence(self, sentence):
        return np.array([self.embed(w) for w in sentence.split()])

    def next_case(self):
        correct = False

        while not correct:
            word = np.random.choice(self.words_list)
            m1, m2 = self.dict1(word), self.dict2(word)
            if (m1 and m2) != None and m1 != []:
                first_key = list(m1.keys())[0]
                if len(m1[first_key]) != 0:
                    correct = True

        m1 = m1[first_key][0]
        m1_embedded = self.embed_sentence(m1)
        m2_embedded = self.embed_sentence(m2)
        return m1_embedded, m2_embedded, self.embed(word), m1, m2, word

    def next_batch(self):
        case = list(zip(*[self.next_case() for i in range(self.batch_size)]))
        x1, x2, y, label = np.array(case[0]), np.array(case[1]), np.array(case[2]), case[3:6]
        max_length1 = np.max([v.shape[0] for v in x1])
        max_length2 = np.max([v.shape[0] for v in x2])
        max_length = np.maximum(max_length1, max_length2)

        def pad(x):
            for i in range(x.shape[0]):
                padding = ((0, max_length - x[i].shape[0]), (0, 0))
                x[i] = np.pad(x[i], padding, 'constant')
            x = np.array([np.array(v) for v in x])
            return x

        return pad(x1), pad(x2), y, label

    def nearest_words(self, vector):
        dists = []
        for other_word in self.words_list:
            other_vector = self.embed(other_word)
            dists.append((cosine_distance(vector, other_vector), other_word))
        return sorted(dists)[:10]

class LoadedDictTask:
    def __init__(self, batch_size):
        # with open(f'data/tasks_150_{batch_size}_{num_batches}_v2.pickle', 'rb') as handle:
            # self.data = pickle.load(handle)
        dict_task = DictTask(batch_size)
        self.data, self.vocab_size, self.word_to_i = dict_task.load_from_file('54k.pickle')
        self.dev_batches = 10 # Batches reserved for dev set.
        self.num_batches = len(self.data) - self.dev_batches - 1
        self.i = 0
        self.j = 1

    def embed(self, word):
        if word not in self.weights.keys():
            self.weights[word] = self.rand_weight()
        return self.weights[word]

    def rand_weight(self):
        dims = len(self.weights['a'])
        return np.random.randn(dims) * .7

    def next_batch(self):
        self.i = self.i + 1 if self.i < self.num_batches else 0
        return self.data[self.i]

    def dev_batch(self):
        self.j = self.j + 1 if self.j < self.dev_batches else 1
        return self.data[-self.j]

    def gen_glove_embeddings(self):
        with open(EMBEDDINGS_PATH, 'rb') as handle:
            self.weights = pickle.load(handle)
        # note: the first vector in the word embeddings is for the empty character.
        #  Thus, we assign a random vector for the empty character. note that
        #  in DictTask.load_from_file we took into account this and added one to the indexes.
        embeddings = np.array([[self.rand_weight()]])
        embeddings = np.array([self.embed(word) for word in self.word_to_i.keys()])
        with open('embeddings_300.pickle', 'wb') as handle:
           pickle.dump(embeddings, handle)

    def glove_embeddings(self):
        with open('embeddings_300.pickle', 'rb') as handle:
            return pickle.load(handle)

def get_words_and_defs():
    data, dict1, dict2 = [], PyDictionary().meaning, pearson_meaning
    words = [x.strip().lower() for x in open(WIKI_WORDS, 'r')][1:]

    for i, word in enumerate(words):
        m1 = dict1(word)
        m2 = dict2(word)
        if (m1 and m2) != None and m1 != []:
            first_key = list(m1.keys())[0]
            if len(m1[first_key]) != 0:
                # print(word)
                data.append((m1, m2, word))

        if i % 200 == 0:
            print (i)
            with open(f'data/words_and_defs_{i}.pickle', 'wb') as handle:
                pickle.dump(data, handle)

def store_tasks(num_batches, batch_size, start=0):
    dict_task = DictTask(batch_size)
    data = []
    if start != 0:
        with open(f'data/tasks_{num_batches}_{batch_size}_{start}.pickle', 'rb') as handle:
            data = pickle.load(handle)
    for i in range(start, num_batches):
        print(f'{i}/{num_batches}')
        data.append(dict_task.next_batch())
        with open(f'data/tasks_{num_batches}_{batch_size}_{i}.pickle', 'wb') as handle:
            pickle.dump(data, handle)

'''
from task import DictTask
dict_task = DictTask(16)
a, b, c = dict_task.next_batch()
a.shape
'''
