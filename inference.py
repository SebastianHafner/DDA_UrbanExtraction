from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms.functional as TF
from utils import geofiles, datasets, experiment_manager, paths, networks, visualization


def run_inference(config_name: str, checkpoint: int):
    cfg = experiment_manager.load_cfg(CONFIG_PATH / f'{config_name}.yaml')

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

        dataset = datasets.UrbanExtractionDataset(cfg, site, no_augmentations=True, include_projection=True)

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



ROOT_PATH = Path('/storage/shafner/urban_extraction')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')
NETWORK_PATH = Path('/storage/shafner/urban_extraction/networks/')


def run_inference(config_name: str, checkpoint: int, s1_file: Path, s2_file: Path, save_dir: Path, name: str):
    # loading config and network
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')
    net_file = NETWORK_PATH / f'{config_name}_{checkpoint}.pkl'
    net = load_network(cfg, net_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    # loading dataset from config (requires inference.json)
    patch_size = 256
    dataset = TilesInferenceDataset(cfg, s1_file=s1_file, s2_file=s2_file, patch_size=patch_size)
    pred = dataset.get_mask('uint8')

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            patch = dataset.__getitem__(i)

            img_patch = patch['x'].to(device)
            i_start, i_end = patch['row']
            j_start, j_end = patch['col']

            activation_patch = net(img_patch.unsqueeze(0))
            activation_patch = torch.sigmoid(activation_patch)
            activation_patch = activation_patch.cpu().detach().numpy()
            activation_patch = activation_patch[0,].transpose((1, 2, 0)).astype('float32')
            pred_patch = activation_patch > cfg.THRESHOLDS.VALIDATION
            pred_patch = pred_patch.astype('uint8')
            rf = dataset.rf
            if i_start == 0 and j_start == 0:
                pred[i_start:i_end - 2 * rf, j_start:j_end - 2 * rf, ] = pred_patch[:-2 * rf, :-2 * rf, ]
            elif i_start == 0:
                pred[i_start:i_end - 2 * rf, j_start + rf:j_end - rf, ] = pred_patch[:-2 * rf, rf:-rf, ]
            elif j_start == 0:
                pred[i_start + rf: i_end - rf, j_start: j_end - 2 * rf, ] = pred_patch[rf:-rf, :-2 * rf, ]
            else:
                pred[i_start + rf: i_end - rf, j_start + rf: j_end - rf, ] = pred_patch[rf:-rf, rf:-rf, ]

        save_dir.mkdir(exist_ok=True)
        pred_file = save_dir / f'pred_{config_name}_{name}.tif'
        write_tif(pred_file, pred, dataset.geotransform, dataset.crs)


if __name__ == '__main__':
    config_name = 'fusion_color'
    checkpoint = 100
    roi_id = 'lagos'

    s1_file = ROOT_PATH / 'inference' / roi_id / f'sentinel1_{roi_id}.tif'
    s2_file = ROOT_PATH / 'inference' / roi_id / f'sentinel2_{roi_id}.tif'

    save_dir = s1_file.parent
    run_inference(config_name, checkpoint, s1_file, s2_file, save_dir, roi_id)

