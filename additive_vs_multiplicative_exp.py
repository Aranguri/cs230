import time
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import copy
import matplotlib.pyplot as plt

training_size = 10000
vector_dim = 100
h1_size, h2_size = 100, 100

vectors = np.random.uniform(-1, 1, (training_size, 1, vector_dim))
vectors = [v / v.std() for v in vectors]
noise = np.random.randn(training_size, 2, vector_dim) * .5

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


model1 = nn.Sequential(
    nn.Linear(2 * vector_dim, h1_size),
    nn.ReLU(),
    nn.Linear(h1_size, h2_size),
    nn.ReLU(),
    nn.Linear(h2_size, 1),
    nn.Sigmoid()
)

model2 = nn.Sequential(
    nn.Linear(3 * vector_dim, h1_size),
    nn.ReLU(),
    nn.Linear(h1_size, h2_size),
    nn.ReLU(),
    nn.Linear(h2_size, 1),
    nn.Sigmoid()
)

loss_acc = np.zeros(100000)
xs = [[], []]
ys = [[], []]


for k in range(2):
    if k == 0: optimizer = optim.Adam(model1.parameters())
    if k == 1: optimizer = optim.Adam(model2.parameters())
    optimizer.zero_grad()

    for j in range(8):
        for i, (input, target) in enumerate(zip(inputs, targets)):
            input = np.array([np.array(v) for v in input])
            v1, v2 = np.split(input, 2, 1)
            v3 = v1 * v2
            if k == 1: input = np.concatenate((input, v3), axis=1)
            target = np.array([np.array(v) for v in target])
            input_ = Variable(torch.from_numpy(input)).float()
            target_ = Variable(torch.from_numpy(target)).float()
            if k == 0: outputs = model1(input_)
            if k == 1: outputs = model2(input_)

            if i != 0:
                loss = nn.functional.binary_cross_entropy(outputs, target_)
                loss.backward()
                optimizer.step()

                loss_acc[i + j * len(inputs)] = loss.item()
            else:
                accuracy = torch.mean((torch.squeeze((outputs > .5).float()) == target_).float())
                temp_loss = loss_acc[:i + j * len(inputs)]
                xs[k].append(i + j * len(inputs))
                ys[k].append(accuracy)#temp_loss[-100].mean())
                print(accuracy)

plt.plot(xs[0], ys[0], label='Model without multiplicative interactions')
plt.plot(xs[1], ys[1], label='Model with multiplicative interactions')
plt.xlabel('Number of batches')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# time.sleep(100000)


#funcan: 21, 14, 12, 5, 4
