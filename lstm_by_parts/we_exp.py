from utils import *
from task import LoadedDictTask, DictTask
import numpy as np

same_sen, diff_sen, n = [], [], 10

for i in range(n):
    #Exp 1: similarity between a name vector and a word from the definition
    dict_task = DictTask(1)
    x, y, _ = dict_task.next_batch()
    x, y = x.squeeze(), y.squeeze()
    for w in x:
        same_sen.append(cosine_distance(w, y))

    #Exp 2: similarity between two random words
    w1, w2 = None, None
    while w1 not in dict_task.weights.keys():
        w1 = np.random.choice(dict_task.words_list)

    while w2 not in dict_task.weights.keys():
        w2 = np.random.choice(dict_task.words_list)

    v1 = dict_task.weights[w1]
    v2 = dict_task.weights[w2]

    diff_sen.append(cosine_distance(v1, v2))

print(np.mean(same_sen))
print(np.mean(diff_sen))

'''
Results:
Same sentence: 0.276916644687
Diff sentence: 0.350012294601
'''

def cosine_distance(v1, v2):
    return np.linalg.norm(v1 - v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
