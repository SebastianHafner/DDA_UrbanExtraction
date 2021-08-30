from pathlib import Path
import torch
from networks.network_loader import load_network
from experiment_manager.config import config
from utils.datasets import GHSLDataset
from utils.geotiff import *
from tqdm import tqdm
import numpy as np
from utils.metrics import *

ROOT_PATH = Path('/storage/shafner/urban_extraction')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')


def merge_patches(site: str, thresh: float = None):
    print(f'running inference for {site}...')

    dataset = GHSLDataset(ROOT_PATH / 'urban_dataset', site, thresh=thresh)
    patch_size = dataset.patch_size

    # config inference directory
    save_path = ROOT_PATH / 'inference' / 'ghsl'
    save_path.mkdir(exist_ok=True)

    ghsl_output = dataset.get_arr()
    transform, crs = dataset.get_geo()

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            patch = dataset.__getitem__(i)
            ghsl_patch = patch['x'].cpu().squeeze().numpy()
            ghsl_patch = np.clip(ghsl_patch, 0, 100).astype('uint8')

            y, x = id2yx(patch['patch_id'])

            ghsl_output[y: y+patch_size, x:x+patch_size, 0] = ghsl_patch

    output_file = save_path / f'ghsl_{site}.tif'
    write_tif(output_file, ghsl_output, transform, crs)


def run_quantitative_evaluation(site: str, thresh: float = 0.5, save_output: bool = False):
    print(f'running inference for {site}...')

    dataset = GHSLDataset(ROOT_PATH / 'urban_dataset', site, label_thresh=0)

    y_probs, y_trues = None, None

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            patch = dataset.__getitem__(i)
            ghsl_prob = patch['x'].cpu().squeeze().flatten()
            label = patch['y'].cpu().flatten()

            if y_probs is not None:
                y_probs = torch.cat((y_probs, ghsl_prob), dim=0)
                y_trues = torch.cat((y_trues, label), dim=0)
            else:
                y_probs = ghsl_prob
                y_trues = label

        if save_output:
            y_probs = y_probs.numpy()
            y_trues = y_trues.numpy()
            output_data = np.stack((y_trues, y_probs))
            output_path = ROOT_PATH / 'quantitative_evaluation' / 'ghsl'
            output_path.mkdir(exist_ok=True)
            output_file = output_path / f'{site}_ghsl.npy'
            np.save(output_file, output_data)
        else:
            y_preds = (y_probs > thresh).float()
            prec = precision(y_trues, y_preds, dim=0)
            rec = recall(y_trues, y_preds, dim=0)
            f1 = f1_score(y_trues, y_preds, dim=0)
            print(f'{site}: f1 score {f1:.3f} - precision {prec:.3f} - recall {rec:.3f}')


if __name__ == '__main__':
    cities_igarss = ['stockholm', 'kampala', 'daressalam', 'sidney']
    cities_igarss = ['newyork', 'sanfrancisco']
    for city in cities_igarss:
        # merge_patches(city)
        run_quantitative_evaluation(city, save_output=True)
