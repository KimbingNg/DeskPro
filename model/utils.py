from utils.logger import global_logger
__model_map = {}

def reg_model(dn):

    def get_dataset(model_class):
        if dn in __model_map:
            raise Exception(f'{dn} 已存在:  {__model_map[dn]}')
        __model_map[dn] = model_class
        return model_class
    return get_dataset

def get_module(dn):
    if dn in __model_map:
        ret = __model_map[dn]
        global_logger.info(f'Load module {ret}')
        return ret
    return None

