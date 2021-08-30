from utils.visualization import *
from pathlib import Path
import numpy as np
import json
import torch
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from networks.network_loader import load_checkpoint
from utils.datasets import SelfsupervisedSpaceNet7Dataset
from experiment_manager.config import config
from utils.metrics import *
from utils.geotiff import *


URBAN_EXTRACTION_PATH = Path('/storage/shafner/urban_extraction')
ROOT_PATH = Path('/storage/shafner/urban_extraction')
DATASET_PATH = Path('/storage/shafner/urban_extraction/urban_dataset')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')
NETWORK_PATH = Path('/storage/shafner/urban_extraction/networks/')

mpl.rcParams.update({'font.size': 20})


def show_quantitative_results(config_name: str):

    # loading config and network
    cfg = config.load_cfg(Path.cwd() / 'configs' / f'{config_name}.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _, _ = load_checkpoint(cfg.INFERENCE.CHECKPOINT, cfg, device)
    net.eval()

    # loading dataset from config (requires inference.json)
    dataset = SelfsupervisedSpaceNet7Dataset(cfg)

    y_preds, y_trues = [], []
    with torch.no_grad():
        for index in tqdm(range(len(dataset))):
            test_site = dataset.__getitem__(index)
            s2_img = test_site['x'].to(device)
            y_pred = net(s2_img.unsqueeze(0))
            y_pred = torch.sigmoid(y_pred).flatten().cpu().numpy()
            y_preds.append(y_pred)
            y_true = test_site['y'].flatten().cpu().numpy()
            y_trues.append(y_true)
            assert(np.size(y_true) == np.size(y_pred))

    y_preds = np.concatenate(y_preds).flatten()
    y_trues = np.concatenate(y_trues).flatten()
    rmse = root_mean_square_error(y_preds, y_trues)
    print(f'RMSE: {rmse:.3f}')


def plot_qualitative_results(config_name: str, save_plots: bool = False):

    # loading config and network
    cfg = config.load_cfg(Path.cwd() / 'configs' / f'{config_name}.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _, _ = load_checkpoint(cfg.INFERENCE.CHECKPOINT, cfg, device)
    net.eval()

    # loading dataset from config (requires inference.json)
    dataset = SelfsupervisedSpaceNet7Dataset(cfg)

    y_preds, y_trues = [], []
    with torch.no_grad():
        for index in tqdm(range(len(dataset))):
            sample = dataset.__getitem__(index)
            aoi_id = sample['aoi_id']
            country = sample['country']
            group_name = sample['group_name']

            fig, axs = plt.subplots(2, 3, figsize=(20, 10))

            optical_file = DATASET_PATH / 'sn7' / 'sentinel2' / f'sentinel2_{aoi_id}.tif'
            plot_optical(axs[0, 0], optical_file, vis='true_color')

            label = cfg.DATALOADER.LABEL
            label_file = DATASET_PATH / 'sn7' / label / f'{label}_{aoi_id}.tif'
            plot_buildings(axs[1, 0], label_file)

            sar_file = DATASET_PATH / 'sn7' / 'sentinel1' / f'sentinel1_{aoi_id}.tif'
            plot_sar(axs[0, 1], sar_file, vis='VV')
            plot_sar(axs[1, 1], sar_file, vis='VH')

            x = sample['x'].to(device)
            logits = net(x.unsqueeze(0))
            sar_pred = torch.sigmoid(logits.squeeze())
            sar_pred = sar_pred.detach().cpu().numpy()
            plot_probability(axs[0, 2], sar_pred[0, ])
            plot_probability(axs[1, 2], sar_pred[1, ])

            if save_plots:
                folder = ROOT_PATH / 'plots' / 'testing' / 'qualitative' / config_name
                folder.mkdir(exist_ok=True)
                file = folder / f'{aoi_id}.png'
                plt.savefig(file, dpi=300, bbox_inches='tight')
            else:
                plt.show()
            plt.close()


if __name__ == '__main__':
    show_quantitative_results('selfsupervised_nogamma')
    plot_qualitative_results('selfsupervised_nogamma', True)
