import os
import numpy as np
import torch as tc
from tensorboardX import SummaryWriter


CURR_PATH = os.path.split(os.path.realpath(__file__))[0]

writer = SummaryWriter(CURR_PATH)
for i in range(10):
    x = np.random.random(10)
    x = tc.softmax(tc.from_numpy(x), dim=0)
    writer.add_histogram('distribution centers', x, i, 'fd')

writer.close()
