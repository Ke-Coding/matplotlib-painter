import os
import numpy as np
import torch as tc
from tensorboardX import SummaryWriter


CURR_PATH = os.path.split(os.path.realpath(__file__))[0]

w = SummaryWriter(log_dir=CURR_PATH)

X = [np.linspace(i, i+2, 11) for i in range(5)]
for i, x in enumerate(X):
    print(f'{i}: {x}')
    # w.add_histogram('a', x, i)
