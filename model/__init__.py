from importlib import import_module

import torch
import torch.nn as nn

from . import utils
from .utils import get_module
from .backbone import *
from .deskpro import deskpro



def make_model(args, ckpt=None):

    device = 'cuda'
    if args.model == 'mae_vit':
        model = mae_vit.get_model(args).to(device)
    elif args.model == 'vit_imagenet':
        model = mae_vit.get_imagenet_vit(args)
    else:
        try:
            module = import_module('model.' + args.model.lower())
            model = getattr(module, args.model)(args).to(device)
        except:
            try:
                import timm
                model = timm.create_model(args.model, args=args)
            except:
                model = get_module(args.model)(args)

    # if not args.cpu and args.nGPU >= 1:
    #     model = nn.DataParallel(model, range(args.nGPU))

    return model
