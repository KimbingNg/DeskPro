import torch
import einops
from torch.nn.parallel import DataParallel
from utils.logger import color_print
from torch import nn

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def load_state_dict(model, path, model_type='model', strict=True, strict_shape=True):
    sd = torch.load(path, map_location='cuda')
    __load_state_dict(model, sd, model_type, strict, strict_shape)

def warning(txt):
    color_print(txt, 'red')

def info(txt):
    color_print(txt, 'green')

def load_model_optim(model, optim, path, strict=True, strict_shape=True):
    sd = torch.load(path, map_location='cuda')
    if 'optim' in sd:
        __load_state_dict(optim, sd, 'optim', strict, strict_shape)
    else:
        warning(f'No "optim" in {path}. Skipped')
    if 'model' in sd:
        __load_state_dict(model, sd, 'model', strict, strict_shape)
    else:
        info(f'No "model" in {path}. Try to load model directly.')
        __load_state_dict(model, sd, 'model', strict, strict_shape)

def __torch_load_state_dict(model, sd, model_type='model', strict=True):
    if model_type == 'optim':
        model.load_state_dict(sd)
    else:
        model.load_state_dict(sd, strict)

def __load_state_dict(model, sd, model_type='model', strict=True, strict_shape=True):
    if strict_shape:
        assert strict, 'Must set strict when strict_shape is set'
    if len(sd.keys()) < 10 and  model_type in sd:
        sd = sd[model_type]
        color_print(f'The state_dict file contains key "{model_type}". Unwrapped.', 'green')

    has_module = False

    for k in sd.keys():
        if k.startswith('module'):
            has_module = True
            break

    if strict_shape:
        if has_module:
            if isinstance(model, DataParallel):
                __torch_load_state_dict(model, sd, model_type, strict)
            else:
                __torch_load_state_dict(model.module, sd, model_type, strict)
            return
        else:
            if isinstance(model, DataParallel):
                __torch_load_state_dict(model.module, sd, model_type, strict)
            else:
                __torch_load_state_dict(model, sd, model_type, strict)
    else:
        s = f'Warning: load `DataParallel` module like parameters to {type(model).__name__}, ' \
            f'and the shape is not restricted !'
        if has_module:
            color_print(s, 'red')
        for k, v in sd.items():
            new_key = k.replace('module.', '') if has_module else k
            if strict:
                assert new_key in model.state_dict()
            try:
                model.state_dict()[k].copy_(v)
            except Exception as e:
                s = f'Warning: Shape mismatch:\n' + str(e.args) + '\n----------'
                color_print(s, 'red')


def list_to_tensor(tensor_list, device='cpu'):
    if isinstance(tensor_list, list):
        return torch.stack([list_to_tensor(l) for l in tensor_list], dim=0)
    else:
        return tensor_list.to(device)

def shape_to_einopsshape(shape):
    return ' '.join(map(lambda x: 'dim' + str(x), shape))

class HookData:

    def __init__(self):
        self.data_dic = {}

    def add_data(self, data):
        if not isinstance(data, torch.Tensor):
            return
        self.shape = shape_to_einopsshape(data.shape[1:])
        device = str(data.device)
        if device in self.data_dic:
            self.data_dic[device].append(data)
        else:
            self.data_dic[device] = [data]

    def raw_data_list(self, transform=None):
        keys = sorted(self.data_dic.keys(), key=lambda x: int(x.split(':')[-1]))
        ret = [self.data_dic[k] for k in keys]
        if transform:
            ret = list(map(transform, ret))
        return ret

    def combine(self, transform=None, toTensor=True):
        temp = self.raw_data_list(transform)
        if toTensor:
            return einops.rearrange(
                list_to_tensor(temp, device='cuda:0'),
                f'ngpu times bs {self.shape} -> times (ngpu bs) {self.shape}'
            )
        else:
            raise NotImplementedError


