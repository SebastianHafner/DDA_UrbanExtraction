import yaml
from pathlib import Path
from utils import experiment_manager

# set the paths
HOME = 'C:/Users/shafner/repos/DDA_UrbanExtraction'  # '/home/shafner/DDA_UrbanExtraction'
DATASET = 'C:/Users/shafner/datasets/urban_dataset'  # '/storage/shafner/urban_extraction/urban_dataset'
DATASET_SPACENET7S1S2 = 'C:/Users/shafner/datasets/spacenet7_s1s2_dataset_v3'
OUTPUT = 'C:/Users/shafner/urban_extraction/output'  # '/storage/shafner/urban_extraction_output/'
SPACENET7_METADATA = 'sn7_metadata_urban_dataset.csv'


# TODO: define return type as cfg node from experiment manager
def load_paths():
    C = experiment_manager.CfgNode()
    C.HOME = HOME
    C.DATASET = DATASET
    C.DATASET_SPACENET7S1S2 = DATASET_SPACENET7S1S2
    C.OUTPUT = OUTPUT
    C.SPACENET7_METADATA = SPACENET7_METADATA
    return C.clone()


def setup_directories():
    dirs = load_paths()

    # inference dir
    inference_dir = Path(dirs.OUTPUT) / 'inference'
    inference_dir.mkdir(exist_ok=True)

    # evaluation dirs
    evaluation_dir = Path(dirs.OUTPUT) / 'evaluation'
    evaluation_dir.mkdir(exist_ok=True)
    quantiative_evaluation_dir = evaluation_dir / 'quantitative'
    quantiative_evaluation_dir.mkdir(exist_ok=True)
    qualitative_evaluation_dir = evaluation_dir / 'qualitative'
    qualitative_evaluation_dir.mkdir(exist_ok=True)

    # testing
    testing_dir = Path(dirs.OUTPUT) / 'testing'
    testing_dir.mkdir(exist_ok=True)

    # saving networks
    networks_dir = Path(dirs.OUTPUT) / 'networks'
    networks_dir.mkdir(exist_ok=True)

    # plots
    plots_dir = Path(dirs.OUTPUT) / 'plots'
    plots_dir.mkdir(exist_ok=True)



if __name__ == '__main__':
    setup_directories()
