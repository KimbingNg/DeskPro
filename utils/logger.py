import logging
import os
import sys
import os.path as osp
import pygments.console
from pathlib import Path

def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if if_train:
            fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='w')
        else:
            fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def color_print(txt, color='green', *args, **kwargs):
    print(pygments.console.colorize(color, txt, ), *args, **kwargs)

def createLogger(name=__name__, enable_file=False):
    from pathlib import Path
    import logging
    import logging.handlers
    from colorlog import ColoredFormatter
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.DEBUG)
    if enable_file:
        log_dir = Path('log') / name
        log_dir.mkdir(exist_ok=True, parents=True)
        default_formatter = logging.Formatter(
            ' %(levelname)-8s  '
            '[ %(asctime)s ]: '
            '- %(filename)s[line:%(lineno)d] - %(message)s'
        )

        def addWachedFileHandler(logger, file=None, level=None, formatter=None, handler_class=None):
            handler_class = handler_class or logging.handlers.WatchedFileHandler
            formatter = formatter or default_formatter
            handler = handler_class(file)
            handler.setLevel(level)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        addWachedFileHandler(logger, log_dir / "debug.log", logging.DEBUG)
        addWachedFileHandler(logger, log_dir / "info.log", logging.INFO, handler_class=DebugFileHandler)
        addWachedFileHandler(logger, log_dir / "error.log", logging.WARNING)


    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    fmt = ColoredFormatter(
        ' %(log_color)s%(levelname)-8s %(reset)s '
        # '[ %(log_color)s%(asctime)s%(reset)s ]: '
        '[ %(log_color)s%(filename)s:%(lineno)d%(reset)s ]: '
        # '%(log_color)s- %(filename)s[line:%(lineno)d] - '
        '%(log_color)s%(message)s %(reset)s'
    )
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)
    return logger


global_logger = createLogger('global_logger')


from torch.utils import tensorboard
writer = None

def create_tensorboard_writer(log_dir):
    global writer
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    writer = tensorboard.SummaryWriter(log_dir=log_dir, flush_secs=60)
    return writer

def get_tensorboard_writer(log_dir=None):
    global writer
    if  writer == None:
        assert log_dir
        writer = create_tensorboard_writer(log_dir)
    return writer
