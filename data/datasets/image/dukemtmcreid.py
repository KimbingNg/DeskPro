from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import glob
import re
#from ..utils import gl
from utils.logger import global_logger

from ..dataset import ImageDataset, ImageDataset_MultiInput
from pathlib import Path
from config import cfg

img_dataset_class = ImageDataset if not cfg.special_input else ImageDataset_MultiInput

class DukeMTMCreID(img_dataset_class):
    """DukeMTMC-reID.

    Reference:
        - Ristani et al. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. ECCVW 2016.
        - Zheng et al. Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro. ICCV 2017.

    URL: `<https://github.com/layumi/DukeMTMC-reID_evaluation>`_
    
    Dataset statistics:
        - identities: 1404 (train + query).
        - images:16522 (train) + 2228 (query) + 17661 (gallery).
        - cameras: 8.
    """
    dataset_dir = 'DukeMTMC-reID'
    dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip'

    def __init__(self, root='', filter_func=None, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        #self.download_dataset(self.dataset_dir, self.dataset_url)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.filter_func = filter_func
        
        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True, filter_func=filter_func)
        query = self.process_dir(self.query_dir, relabel=False, filter_func=None)
        gallery = self.process_dir(self.gallery_dir, relabel=False, filter_func=None)

        super(DukeMTMCreID, self).__init__(train, query, gallery, **kwargs)

    def resolve(self, file_path):
        file_name = Path(file_path).name
        pattern = re.compile(r'([-\d]+)_c(\d*)_(\d*)')
        #pid, camid, frame_id = map(int, pattern.search(file_name).groups())
        pid, camid, frame_id = pattern.search(file_name).groups()
        return pid, camid, frame_id

    def process_dir(self, dir_path, relabel=False, filter_func=None):
        img_paths = list(glob.glob(osp.join(dir_path, '*.jpg')))
        if filter_func:
            img_paths = list(filter(filter_func, img_paths))

        pid_container = set()
        for img_path in img_paths:
            try:
                pid, _, _ = self.resolve(img_path)
                #pid, _ = map(int, pattern.search(img_path).groups())
                pid_container.add(pid)
            except:
                print(img_path)
                raise

        pid2label = {pid:label for label, pid in enumerate(sorted(pid_container))}

        data = []
        for img_path in img_paths:
            pid, camid, _ = self.resolve(img_path)
            #assert 1 <= camid <= 9, f'camid {camid} must between 1, 8'
            #camid -= 1 # index starts from 0
            if relabel: pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data

class DukeMTMCreID_MultiInput(DukeMTMCreID):

    def find_face(self, train_test_val, img_file_name):
        raise NotImplementedError()

    def process_dir(self, dir_path, relabel=False, filter_func=None):
        img_paths = list(glob.glob(osp.join(dir_path, '*.jpg')))
        if filter_func:
            old_len = len(img_paths)
            img_paths = list(filter(filter_func, img_paths))
            new_len = len(img_paths)
            global_logger.info(f'Filter out {old_len - new_len} images from the dataset for training.')

        pid_container = set()
        for img_path in img_paths:
            try:
                pid, _, _ = self.resolve(img_path)
                pid_container.add(pid)
            except:
                print(img_path)
                raise

        pid2label = {pid:label for label, pid in enumerate(sorted(pid_container))}

        data = []
        for img_path in img_paths:
            pid, camid, _ = self.resolve(img_path)
            train_test_val = Path(img_path).parent.name
            face_img_path = self.find_face(train_test_val, Path(img_path).name)
            img_path = [img_path] + face_img_path
            #assert 1 <= camid <= 9, f'camid {camid} must between 1, 8'
            #camid -= 1 # index starts from 0
            if relabel: pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data
