from utils.visualization import *
from pathlib import Path
import numpy as np
import torch
from networks.network_loader import load_network
from utils.dataloader import UrbanExtractionDataset
from experiment_manager.config import config

DATASET_PATH = Path('/storage/shafner/urban_extraction/urban_extraction')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')
NETWORK_PATH = Path('/storage/shafner/urban_extraction/networks/')


def random_selection(config_name: str, site: str, n: int):

    cfg_file = CONFIG_PATH / f'{config_name}.yaml'
    cfg = config.load_cfg(cfg_file)

    # loading dataset
    dataset = UrbanExtractionDataset(cfg, site, no_augmentations=True)

    # loading network
    net_file = NETWORK_PATH / f'{config_name}.pkl'
    net = load_network(cfg, net_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    item_indices = list(np.random.randint(0, len(dataset), size=n))

    for index in item_indices:
        sample = dataset.__getitem__(index)
        patch_id = sample['patch_id']

        fig, axs = plt.subplots(2, 3, figsize=(10, 6))

        optical_file = DATASET_PATH / site / 'sentinel2' / f'sentinel2_{site}_{patch_id}.tif'
        plot_optical(axs[0, 0], optical_file, show_title=True)
        plot_optical(axs[0, 1], optical_file, vis='false_color', show_title=True)

        sar_file = DATASET_PATH / site / 'sentinel1' / f'sentinel1_{site}_{patch_id}.tif'
        plot_sar(axs[0, 2], sar_file, show_title=True)


        label = cfg.DATALOADER.LABEL
        label_file = DATASET_PATH / site / label / f'{label}_{site}_{patch_id}.tif'
        plot_buildings(axs[1, 0], label_file, show_title=True)

        with torch.no_grad():
            x = sample['x'].to(device)
            logits = net(x.unsqueeze(0))
            prob = torch.sigmoid(logits[0, 0, ])
            prob = prob.detach().cpu().numpy()
            pred = prob > cfg.THRESH

            plot_activation(axs[1, 2], prob, show_title=True)
            plot_prediction(axs[1, 1], pred, show_title=True)

        plt.show()


def performance_selection(config_name: str, site: str, n: int, reverse: bool = False):

    cfg_file = CONFIG_PATH / f'{config_name}.yaml'
    cfg = config.load_cfg(cfg_file)

    # loading dataset
    dataset = UrbanExtractionDataset(cfg, site, no_augmentations=True)

    # loading network
    net_file = NETWORK_PATH / f'{config_name}.pkl'
    net = load_network(cfg, net_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    # getting patch ids of worst performing ones
    results = []
    with torch.no_grad():
        # TODO: add progress bar
        for index in range(len(dataset)):
            sample = dataset.__getitem__(index)
            x = sample['x'].to(device)
            y_logits = net(x.unsqueeze(0))
            y_prob = torch.sigmoid(y_logits[0, ])
            y_pred = y_prob > cfg.THRESH

            y_true = sample['y'].to(device)
            n_correct = torch.sum(y_pred == y_true)
            results.append((index, n_correct.item()))

    results_sorted = sorted(results, key=lambda x: x[1], reverse=reverse)
    for index, _ in results_sorted[:n]:
        sample = dataset.__getitem__(index)
        patch_id = sample['patch_id']

        fig, axs = plt.subplots(1, 4, figsize=(10, 4))
        for ax in axs:
            ax.set_axis_off()

        sar_file = DATASET_PATH / site / 'sentinel1' / f'sentinel1_{site}_{patch_id}.tif'
        plot_sar(axs[0], sar_file)

        optical_file = DATASET_PATH / site / 'sentinel2' / f'sentinel2_{site}_{patch_id}.tif'
        plot_optical(axs[0], optical_file)

        label = cfg.DATALOADER.LABEL
        label_file = DATASET_PATH / site / label / f'{label}_{site}_{patch_id}.tif'
        plot_buildings(axs[1], label_file)

        with torch.no_grad():
            x = sample['x'].to(device)
            logits = net(x.unsqueeze(0))
            prob = torch.sigmoid(logits[0, 0,])
            pred = prob > cfg.THRESH
            pred = pred.detach().cpu().numpy()
            cmap = colors.ListedColormap(['white', 'red'])
            boundaries = [0, 0.5, 1]
            norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
            axs[2].imshow(pred, cmap=cmap, norm=norm)

        plt.show()


def input_comparison(config_names: list, site: str, n: int):
    pass
    # TODO: to be implemented


if __name__ == '__main__':
    config_name = 'baseline_optical'
    random_selection(config_name, 'montreal', 20)
    # performance_selection(config_name, 'calgary', 10, reverse=True)
