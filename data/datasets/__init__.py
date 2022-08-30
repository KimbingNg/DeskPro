from __future__ import absolute_import
from __future__ import print_function

from .dataset import Dataset, ImageDataset, VideoDataset
from .image import *
from utils.logger import global_logger
from .image import util as img_util


__image_datasets = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
}



def init_image_dataset(name, **kwargs):
    """Initializes an image dataset."""
    try:
        avai_datasets = list(__image_datasets.keys())
        if name not in avai_datasets:
            raise ValueError('Invalid dataset name. Received "{}", '
                             'but expected to be one of {}'.format(name, avai_datasets))
        global_logger.info(f'Load dataset {name}: {__image_datasets[name]}')
        return __image_datasets[name](**kwargs)
    except:
        return img_util.get_dataset(name)(**kwargs)



def init_video_dataset(name, **kwargs):
    """Initializes a video dataset."""
    avai_datasets = list(__video_datasets.keys())
    if name not in avai_datasets:
        raise ValueError('Invalid dataset name. Received "{}", '
                         'but expected to be one of {}'.format(name, avai_datasets))
    return __video_datasets[name](**kwargs)


def register_image_dataset(name, dataset):
    """Registers a new image dataset.

    Args:
        name (str): key corresponding to the new dataset.
        dataset (Dataset): the new dataset class.

    Examples::
        
        import torchreid
        import NewDataset
        torchreid.data.register_image_dataset('new_dataset', NewDataset)
        # single dataset case
        datamanager = torchreid.data.ImageDataManager(
            root='reid-data',
            sources='new_dataset'
        )
        # multiple dataset case
        datamanager = torchreid.data.ImageDataManager(
            root='reid-data',
            sources=['new_dataset', 'dukemtmcreid']
        )
    """
    global __image_datasets
    curr_datasets = list(__image_datasets.keys())
    if name in curr_datasets:
        raise ValueError('The given name already exists, please choose '
                         'another name excluding {}'.format(curr_datasets))
    __image_datasets[name] = dataset


def register_video_dataset(name, dataset):
    """Registers a new video dataset.

    Args:
        name (str): key corresponding to the new dataset.
        dataset (Dataset): the new dataset class.

    Examples::
        
        import torchreid
        import NewDataset
        torchreid.data.register_video_dataset('new_dataset', NewDataset)
        # single dataset case
        datamanager = torchreid.data.VideoDataManager(
            root='reid-data',
            sources='new_dataset'
        )
        # multiple dataset case
        datamanager = torchreid.data.VideoDataManager(
            root='reid-data',
            sources=['new_dataset', 'ilidsvid']
        )
    """
    global __video_datasets
    curr_datasets = list(__video_datasets.keys())
    if name in curr_datasets:
        raise ValueError('The given name already exists, please choose '
                         'another name excluding {}'.format(curr_datasets))
    __video_datasets[name] = dataset
