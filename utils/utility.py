import os
import datetime

import shutil
import matplotlib
from pathlib import Path
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import os.path as osp
import errno

import yaml
from collections import OrderedDict
from config import cmd_args
from shutil import copyfile, copytree
import pickle
import warnings
from optim import make_optimizer, make_scheduler
#from aim import Run
from utils import logger
from utils.logger import global_logger


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.log = torch.Tensor()
        self.since = datetime.datetime.now()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        self.dir = args.save
        Path(self.dir).mkdir(exist_ok=True, parents=True)

        if os.path.exists(self.dir + '/map_log.pt'):
            self.log = torch.load(self.dir + '/map_log.pt')

        global_logger.info('Experiment results will be saved in {} '.format(self.dir))

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)

        ######### For Neptune: ############

        with open(self.dir + '/config.yaml', open_type) as fp:
            fp.write('='*20)
            fp.write('\n')
            fp.write(f'{datetime.datetime.now().strftime("%m%d_%H-%M-%S")}\n')
            fp.write('='*20)
            fp.write('\n')

            dic = args.copy()
            yaml.dump(dic, fp, default_flow_style=False)


    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, epoch=0, refresh=False, end='\n'):
        time_elapsed = (datetime.datetime.now() - self.since).seconds
        log = log + ' Time used: {} m {} s'.format(
            time_elapsed // 60, time_elapsed % 60)
        print(log, end=end)
        if end != '':
            self.log_file.write(log + end)
            try:
                t = log.find('Total')
                m = log.find('mAP')
                r = log.find('rank1')
                writer = logger.get_tensorboard_writer()
                dataset_name = self.args.data_train.split('_')[0]
                writer.add_scalar(f'{dataset_name}/train/batch_loss', float(log[t + 7:t + 12]), epoch) if t > -1 else None
                writer.add_scalar(f'{dataset_name}/test/mAP', float(log[m + 5:m + 11]), epoch) if m > -1 else None
                writer.add_scalar(f'{dataset_name}/test/rank1', float(log[r + 7:r + 13]), epoch) if r > -1 else None
                writer.flush()
            except Exception as e:
                global_logger.warning(e)

        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

            # For Google Drive
            #copyfile(self.dir + '/log.txt', self.dir +
                     #'/log.txt') if self.dir is not None else None

    def done(self):
        self.log_file.close()

    def plot_map_rank(self, epoch):
        axis = np.linspace(1, epoch, self.log.size(0))
        label = 'Reid on {}'.format(self.args.data_test)
        labels = ['mAP', 'rank1', 'rank3', 'rank5', 'rank10']
        fig = plt.figure()
        plt.title(label)
        for i in range(len(labels)):
            plt.plot(axis, self.log[:, i + 1].numpy(), label=labels[i])

        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('mAP/rank')
        plt.grid(True)
        plt.savefig('{}/result_{}.pdf'.format(self.dir,
                                              self.args.data_test), dpi=600)
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        pass

    def save_checkpoint(
        self, state, save_dir, file_tag='', is_best=False, remove_module_from_keys=False
    ):
        r"""Saves checkpoint.

        Args:
            state (dict): dictionary.
            save_dir (str): directory to save checkpoint.
            is_best (bool, optional): if True, this checkpoint will be copied and named
                ``model-best.pth.tar``. Default is False.
            remove_module_from_keys (bool, optional): whether to remove "module."
                from layer names. Default is False.

        Examples::
            >>> state = {
            >>>     'state_dict': model.state_dict(),
            >>>     'epoch': 10,
            >>>     'rank1': 0.5,
            >>>     'optimizer': optimizer.state_dict()
            >>> }
            >>> save_checkpoint(state, 'log/my_model')
        """
        Path(save_dir).mkdir(exist_ok=True, parents=True)

        if remove_module_from_keys:
            # remove 'module.' in state_dict's keys
            state_dict = state['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                new_state_dict[k] = v
            state['state_dict'] = new_state_dict

        fpath = osp.join(save_dir, f'{file_tag}.pth' if file_tag else 'model-latest.pth')
        torch.save(state, fpath)
        global_logger.info('[INFO] Checkpoint saved to "{}"'.format(fpath))

        if is_best:
            shutil.copy(fpath, (Path(fpath).parent / 'model-best.pth.tar').resolve().as_posix())
                        #osp.join(osp.dirname(fpath), 'model-best.pth.tar'))
        if 'log' in state.keys():
            torch.save(state['log'], os.path.join(save_dir, 'map_log.pt'))

    def load_checkpoint(self, fpath):
        """Loads checkpoint.
        ``UnicodeDecodeError`` can be well handled, which means
        python2-saved files can be read from python3.
        Args:
            fpath (str): path to checkpoint.
        Returns:
            dict
        Examples::
            >>> fpath = 'log/my_model/model.pth.tar-10'
            >>> checkpoint = load_checkpoint(fpath)
        """
        if fpath is None:
            raise ValueError('File path is None')
        if not osp.exists(fpath):
            raise FileNotFoundError('File is not found at "{}"'.format(fpath))
        #map_location = None if torch.cuda.is_available() else 'cpu'
        map_location = 'cpu'
        try:
            checkpoint = torch.load(fpath, map_location=map_location)
        except UnicodeDecodeError:
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            checkpoint = torch.load(
                fpath, pickle_module=pickle, map_location=map_location
            )
        except Exception:
            print('Unable to load checkpoint from "{}"'.format(fpath))
            raise
        return checkpoint

    def load_weight(self, model, weight_path):
        r"""Loads pretrianed weights to model.
        Features::
            - Incompatible layers (unmatched in name or size) will be ignored.
            - Can automatically deal with keys containing "module.".
        Args:
            model (nn.Module): network model.
            weight_path (str): path to pretrained weights.
        """
        checkpoint = self.load_checkpoint(weight_path)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if not cmd_args.resume or cmd_args.test:
            # 测试的时候  或者 非 resume  (pre_train)
            # for k, v in state_dict.copy().items():
            #     if 'reduction' in k and 'classifier' in k:
            #         del state_dict[k]
            #     if 'face_net.fc' in k:
            #         del state_dict[k]
                #if 'face_net' in k:
                    #del state_dict[k]
            strict = False
        else:
            strict = True
        #print(state_dict.keys())
        model.load_state_dict(state_dict, strict=strict)
        if not strict:
            global_logger.warning(f"loading model {type(model)}, strict = False")
        return

    def resume_from_checkpoint(self, fpath, model, optimizer=None, scheduler=None, args=None):
        r"""Resumes training from a checkpoint.

        This will load (1) model weights and (2) ``state_dict``
        of optimizer if ``optimizer`` is not None.

        Args:
            fpath (str): path to checkpoint.
            model (nn.Module): model.
            optimizer (Optimizer, optional): an Optimizer.
            scheduler (LRScheduler, optional): an LRScheduler.

        Returns:
            int: start_epoch.

        Examples::
            >>> from torchreid.utils import resume_from_checkpoint
            >>> fpath = 'log/my_model/model.pth.tar-10'
            >>> start_epoch = resume_from_checkpoint(
            >>>     fpath, model, optimizer, scheduler
            >>> )
        """
        checkpoint = self.load_checkpoint(fpath)

        self.load_weight(model, fpath)
        model = model.cuda()
        if optimizer is not None and 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            global_logger.info('[INFO] Optimizer loaded')
        start_epoch = checkpoint['epoch']
        if scheduler is None:
            scheduler = make_scheduler(args, optimizer, start_epoch) # 需要 optimizer 设置好 initial_lr
        if scheduler is not None and 'scheduler' in checkpoint.keys():
            scheduler.load_state_dict(checkpoint['scheduler'])
            global_logger.info('[INFO] Scheduler loaded')
        global_logger.info(f'Checkpoint saved: [epoch]: {start_epoch}')
        if 'rank1' in checkpoint.keys():
            global_logger.info(
                '[INFO] Last rank1 = {}'.format(checkpoint['rank1']))
        if 'log' in checkpoint.keys():
            self.log = checkpoint['log']


        return start_epoch, model, optimizer, scheduler
