import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import copy

training_size = 10000
vector_dim = 100
h1_size, h2_size = 300, 300

vectors = np.random.uniform(-1, 1, (training_size, 1, vector_dim))
vectors /= vectors.std()
noise = np.random.randn(training_size, 2, vector_dim)

noisy_vectors = vectors + noise
correct_vectors = noisy_vectors.reshape(training_size, 2 * vector_dim)
incorrect_vectors = copy.deepcopy(noisy_vectors.reshape(2 * training_size, vector_dim))
np.random.shuffle(incorrect_vectors)
incorrect_vectors = incorrect_vectors.reshape(training_size, 2 * vector_dim)

correct_tuples = [(v, 1) for v in correct_vectors]
incorrect_tuples = [(v, 0) for v in incorrect_vectors]
input_tuples = np.concatenate((correct_tuples, incorrect_tuples), 0)
np.random.shuffle(input_tuples)

inputs, targets = input_tuples.reshape(100, 200, 2).T

model = nn.Sequential(
    nn.Linear(2 * vector_dim, h1_size),
    nn.ReLU(),
    nn.Linear(h1_size, h2_size),
    nn.ReLU(),
    nn.Linear(h2_size, 1),
    nn.Sigmoid()
)

loss_acc = np.zeros(100000)
for j in range(1000):
    for i, (input, target) in enumerate(zip(inputs, targets)):
        optimizer = optim.Adam(model.parameters())
        optimizer.zero_grad()

        input = np.array([np.array(v) for v in input])
        v1, v2 = np.split(input, 2, 1)
        v3 = v1 * v2
        #input = np.concatenate((input, v3), axis=1)
        target = np.array([np.array(v) for v in target])
        input_ = Variable(torch.from_numpy(input)).float()
        target_ = Variable(torch.from_numpy(target)).float()
        outputs = model(input_)

        loss = nn.functional.binary_cross_entropy(outputs, target_)
        accuracy = torch.mean(((outputs > .5).float() == target_).float())
        loss.backward()
        optimizer.step()

        loss_acc[i + j * len(inputs)] = loss.item()
        if i % 101 == 100:
            temp_loss = loss_acc[:i + j * len(inputs)]
            print(i + j * len(inputs), temp_loss[-100].mean())

#funcan: 21, 14, 12, 5, 4
