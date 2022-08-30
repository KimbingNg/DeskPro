from utils.logger import global_logger
__dataset_map = {}

def register_dataset(dn):

    def get_dataset(model_class):
        if dn in __dataset_map:
            raise Exception(f'数据集  {dn} 已存在:   {__dataset_map[dn]}')
        __dataset_map[dn] = model_class
        return model_class
    return get_dataset

def get_dataset(dn):
    if dn in __dataset_map:
        ret = __dataset_map[dn]
        global_logger.info(f'Load dataset {ret}')
        return ret
    return None

