import argparse
import logging
from argparse import ArgumentParser
from tabulate import tabulate
from collections import OrderedDict
import yaml
from fvcore.common.config import CfgNode as _CfgNode
from pathlib import Path


def default_argument_parser():

    parser = argparse.ArgumentParser(description="Experiment Args")
    parser.add_argument('-c', "--config-file", dest='config_file', default="", required=True, metavar="FILE",
                        help="path to config file")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


class CfgNode(_CfgNode):
    """
    The same as `fvcore.common.config.CfgNode`, but different in:

    1. Use unsafe yaml loading by default.
      Note that this may lead to arbitrary code execution: you must not
      load a config file from untrusted sources before manually inspecting
      the content of the file.
    2. Support config versioning.
      When attempting to merge an old config, it will convert the old config automatically.

    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Always allow merging new configs
        self.__dict__[CfgNode.NEW_ALLOWED] = True
        super(CfgNode, self).__init__(init_dict, key_list, True)

    # Note that the default value of allow_unsafe is changed to True
    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = True) -> None:
        loaded_cfg = _CfgNode.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)

        # defaults.py needs to import CfgNode
        self.merge_from_other_cfg(loaded_cfg)


def new_config():
    '''
    Creates a new config based on the default config file
    :return:
    '''

    C = CfgNode()

    C.CONFIG_DIR = 'config/'
    C.OUTPUT_BASE_DIR = 'output/'

    # TRAINER SETTINGS
    C.TRAINER = CfgNode()
    C.TRAINER.LR = 0.001
    C.TRAINER.BATCH_SIZE = 1
    C.TRAINER.CHECKPOINT_PERIOD = 5000
    C.TRAINER.EPOCHS = 1

    # DATALOADER SETTINGS
    C.DATALOADER = CfgNode()
    C.DATALOADER.NUM_WORKER = 1
    C.DATALOADER.SHUFFLE = True

    # DATASET SETTINGS
    C.DATASETS = CfgNode()
    C.DATASETS.TRAIN = ()
    C.DATASETS.TEST = ()

    # Model configs
    C.MODEL = CfgNode()
    C.MODEL.BINARY_CLASSIFICATION = False
    C.MODEL.OUT_CHANNELS = 1
    C.MODEL.IN_CHANNELS = 3

    C.MAX_EPOCHS = 1
    return C.clone()


# loading cfg
def load_cfg(yaml_file: Path):
    cfg = new_config()
    cfg.merge_from_file(str(yaml_file))
    cfg.NAME = str(yaml_file.stem)
    return cfg


def setup_cfg(args):
    cfg = new_config()
    cfg.merge_from_file(f'configs/{args.config_file}.yaml')
    cfg.merge_from_list(args.opts)
    cfg.NAME = args.config_file
    return cfg
