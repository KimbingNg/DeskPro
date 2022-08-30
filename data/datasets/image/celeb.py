# encoding: utf-8
from .dukemtmcreid import DukeMTMCreID, DukeMTMCreID_MultiInput
from .util import  register_dataset
from config import cfg
from utils.logger import global_logger
from pathlib import Path
import re

parent_class = DukeMTMCreID

if cfg.special_input:
    parent_class = DukeMTMCreID_MultiInput


def assert_filter(assert_exist_list):
    def name_to_name(file_name):
        return file_name

    def filter_func(img_path):
        for ae in assert_exist_list:
            img_path = Path(img_path)
            train_test_val = img_path.parent.name
            if not (Path(ae) / train_test_val / name_to_name(img_path.name)).exists():
                return False
        return True
    return filter_func

lr_face_dir = cfg.dataset_path_cfg.celeb.lr_face_dir
face_dir = cfg.dataset_path_cfg.celeb.face_dir
clothing_mask_dir = cfg.dataset_path_cfg.celeb.clothing_mask_dir

def find_face(train_test_val, img_file_name):
    bace_dir = f'{face_dir}/{train_test_val}'
    face_img_path = Path(bace_dir) / img_file_name
    return [str(face_img_path)]


@register_dataset('celeb_lr_hr_mask')
class CELEB_LR_HR_MASK(parent_class):
    def find_face(self, train_test_val, img_file_name):
        ret = []
        for d in self.dirs:
            bace_dir = f'{d}/{train_test_val}'
            img_path = Path(bace_dir) / img_file_name
            ret.append(str(img_path))
        return ret

    def __init__(self, root='', prcc_type='fg-cc', **kwargs):
        p = Path(cfg.dataset_path_cfg.celeb.img_dir)
        root, self.dataset_dir = str(p.parent), p.name
        self.dirs = [lr_face_dir, face_dir, clothing_mask_dir]
        super(CELEB_LR_HR_MASK, self).__init__(root, assert_filter(self.dirs), **kwargs)

        assert self.mode in ['train', 'query', 'gallery']
