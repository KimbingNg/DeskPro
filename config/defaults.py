# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
import sys

from yacs.config import CfgNode as CN
from .option import args
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
command_line = ' '.join(map(lambda x: repr(x), sys.argv))
import torch, random
import numpy as np
from utils.logger import global_logger

cfg = CN()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def configure():
    global cfg, args

    if args.config != "":
        with open('config_files/base.yaml', 'r') as f:
            cfg = cfg.load_cfg(f)
        cfg.merge_from_file(args.config)

    cfg.merge_from_list(args.opts)


    import math
    if 'prcc' in cfg.data_train:
        # batch_base = 40 * 8
        batch_base = 20
    else:
        batch_base = 20
    cfg.lr = math.sqrt((cfg.batchid * cfg.batchimage) / batch_base) * 0.0006
    global_logger.warning(f'learning rate {cfg.lr}')

    cfg.test_only = args.test


configure()

cmd_args = args
torch.set_num_threads(8)