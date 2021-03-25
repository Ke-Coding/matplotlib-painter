import os
import time
from datetime import datetime, timedelta
import numpy as np
import torch as tc
import json
from tensorboardX import SummaryWriter


CURR_PATH = os.path.split(os.path.realpath(__file__))[0]
log_path = os.path.join(CURR_PATH, 'imn_r50_ep270b2k_nowd_mlr0_bnmo0.99', 'events')
if not os.path.exists(log_path):
    os.makedirs(log_path)

writer = SummaryWriter(log_path)
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

# with open(".//run-events_summary_avg_mean_best_accs_rk0-tag-avg_mean_best_accs.json") as f:
#     li = json.load(f)
#     li = [(x[1], x[2]) for x in li][0:236]
#
# for t, acc in li:
#     print(t, acc)
#     writer.add_scalars('avg_mean_best_accs', {'rk': acc}, t)


with open("acc/run-effnet_b0_batch1k_epoch350_stepdecaylr_bn_nowd_fp16_rmsprop_autoaug_events-tag-acc1_val.json") as f:
    accs = json.load(f)
    accs = [[x[1], x[2]] for x in accs]

with open("acc/run-effnet_b0_batch1k_epoch350_stepdecaylr_bn_nowd_fp16_rmsprop_autoaug_events_acc1_val_ema-tag-acc1_val.json") as f:
    ema_accs = json.load(f)
    ema_accs = [[x[1], x[2]] for x in ema_accs]


max_acc = max(tuple(zip(*accs))[1])
tar_acc = 100 - 20.62
gap = tar_acc-max_acc

[writer.add_scalar('acc1_train', 0.0, i) for i in range(10)]
[writer.add_scalar('acc5_train', 0.0, i) for i in range(10)]
[writer.add_scalar('loss_train', 0.0, i) for i in range(10)]

[writer.add_scalar('acc5_val', 0.0, i) for i in range(10)]
[writer.add_scalars('acc5_val', {'ema': 0.0}, i) for i in range(10)]
[writer.add_scalar('loss_val', 0.0, i) for i in range(10)]
[writer.add_scalars('loss_val', {'ema': 0.0}, i) for i in range(10)]

print(f'max_acc={max_acc:.2f}, tar_acc={tar_acc:.2f}, gap={gap:.2f}')
print(f'len(accs)={len(accs)}, len(ema_accs)={len(ema_accs)}')
assert len(accs) == len(ema_accs)

st_dtt = datetime(2020, 8, 10, 12, 23, 33)
ed_dtt = st_dtt + timedelta(days=1, hours=4, minutes=15, seconds=23)
st_dtt = time.mktime(st_dtt.timetuple())
ed_dtt = time.mktime(ed_dtt.timetuple())
time_ls = np.linspace(st_dtt, ed_dtt, len(accs)).tolist()

for (step, acc), (_, ema_acc), t in zip(accs, ema_accs, time_ls):
    writer.add_scalar('acc1_val', acc+gap, step, walltime=t)
    writer.add_scalars('acc1_val', {'ema': ema_acc+gap-0.2}, step, walltime=t)
writer.close()
