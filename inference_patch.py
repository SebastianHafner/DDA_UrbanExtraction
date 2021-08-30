from utils.visualization import *
from pathlib import Path
import numpy as np
import torch
from networks.network_loader import load_network
from utils.dataloader import UrbanExtractionDataset, SpaceNet7Dataset
from experiment_manager.config import config
from tqdm import tqdm

DATASET_PATH = Path('/storage/shafner/urban_extraction/urban_extraction_dataset')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')
NETWORK_PATH = Path('/storage/shafner/urban_extraction/networks/')


def run_inference(config_name: str, checkpoint: int):
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')

    # loading network
    net_file = NETWORK_PATH / f'{config_name}_{checkpoint}.pkl'
    net = load_network(cfg, net_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    sites = cfg.DATASETS.SITES.TRAINING + cfg.DATASETS.SITES.VALIDATION
    for site in tqdm(sites):
        # create save folder
        prediction_name = f'prediction_{config_name}'
        prediction_folder = DATASET_PATH / site / prediction_name
        prediction_folder.mkdir(exist_ok=True)

        dataset = UrbanExtractionDataset(cfg, site, no_augmentations=True, include_projection=True)

        for index in range(len(dataset)):
            sample = dataset.__getitem__(index)
            patch_id = sample['patch_id']
            transform = sample['transform']
            crs = sample['crs']
            file_name = f'{prediction_name}_{site}_{patch_id}.tif'
            file = prediction_folder / file_name

            with torch.no_grad():
                x = sample['x'].to(device)
                net_output = net(x.unsqueeze(0))
                prob = torch.sigmoid(net_output.squeeze().unsqueeze(-1))
                prob = prob.detach().cpu().numpy()

            write_tif(file, prob, transform, crs)


def run_inference_sn7(config_name: str, checkpoint: int):
    cfg_file = CONFIG_PATH / f'{config_name}.yaml'
    cfg = config.load_cfg(cfg_file)

    # loading network
    net_file = NETWORK_PATH / f'{config_name}_{checkpoint}.pkl'
    net = load_network(cfg, net_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    dataset = SpaceNet7Dataset(cfg)
    # create save folder
    prediction_name = f'prediction_{config_name}'
    prediction_folder = DATASET_PATH / 'sn7' / prediction_name
    prediction_folder.mkdir(exist_ok=True)

    for index in tqdm(range(len(dataset))):
        sample = dataset.__getitem__(index)
        aoi_id = sample['aoi_id']
        transform = sample['transform']
        crs = sample['crs']
        file_name = f'{prediction_name}_{aoi_id}.tif'
        file = prediction_folder / file_name

        with torch.no_grad():
            x = sample['x'].to(device)
            net_output = net(x.unsqueeze(0))
            prob = torch.sigmoid(net_output.squeeze().unsqueeze(-1))
            prob = prob.detach().cpu().numpy()

        write_tif(file, prob, transform, crs)


if __name__ == '__main__':
    config_name = 'sar_gamma_smallnet'
    checkpoint = 100
    run_inference(config_name, checkpoint)
    run_inference_sn7(config_name, checkpoint)
