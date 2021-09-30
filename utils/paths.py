import yaml
from pathlib import Path
from utils.experiment_manager import CfgNode

# set the paths
HOME = '/home/shafner/DA_UrbanExtraction'
DATASET = '/storage/shafner/urban_extraction/urban_dataset'
OUTPUT = '/storage/shafner/urban_change_detection/'
SPACENET7_METADATA = 'sn7_metadata_urban_dataset.csv'


def load_paths() -> dict:
    C = CfgNode()
    C.HOME = HOME
    C.DATASET = DATASET
    C.OUTPUT = OUTPUT
    C.SPACENET7_METADATA

    return C.clone()
