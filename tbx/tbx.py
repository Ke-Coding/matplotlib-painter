import os
import numpy as np
import torch as tc
from tensorboardX import SummaryWriter


CURR_PATH = os.path.split(os.path.realpath(__file__))[0]

writer = SummaryWriter(CURR_PATH)
# x = np.array([-0.5, -0.2, 0, 0.2, 0.5])
# x = tc.sigmoid(tc.from_numpy(x))
# writer.add_histogram('xx', x, 0, 'fd')
#
# x = np.array([0.5, 0.2, 0, -0.2, -0.5])
# x = (tc.from_numpy(x))
# writer.add_histogram('xx', x, 1, 'fd')
#
#
# x = np.array([0.5, 0.5, 0.1, 0.5, 0.5])
# x = (tc.from_numpy(x))
# writer.add_histogram('xx', x, 2, 'fd')
#
#
# x = np.array([-0.5, -0.5, 1, -0.5, -0.5])
# x = (tc.from_numpy(x))
# writer.add_histogram('xx', x, 3, 'fd')

writer.add_text('exp_dir', f'~/{os.path.relpath(os.getcwd(), os.path.expanduser("~"))}')

writer.close()
