import random
import numpy as np
import pickle

from tqdm import tqdm

lines = open('HI-Small_Trans.csv').readlines()[1:]
random.shuffle(lines)

positives, negatives = [], []

for line in tqdm(lines):
    line = line.strip().split(',')
    line[0] = line[0].strip().split()[-1]
    if line[-1] == '0':
        positives.append(line)
    else:
        negatives.append(line)

print(len(positives))
print(len(negatives))

total = []

inds = np.random.choice(len(positives), size = 30000)
for i in inds:
    total.append(positives[i])

total.extend(negatives)

random.shuffle(total)

num_train = int(len(total) * 0.8)

train, test = total[:num_train], total[num_train:]

print('total train number: %d' %len(train))
print('total test number: %d' %len(test))

with open('train.pkl', 'wb') as fout:
    pickle.dump(train, fout)

with open('test.pkl', 'wb') as fout:
    pickle.dump(test, fout)


