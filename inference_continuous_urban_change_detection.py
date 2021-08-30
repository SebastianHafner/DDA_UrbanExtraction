from pathlib import Path
import torch
from experiment_manager.config import config
from networks.network_loader import load_checkpoint
from utils import geotiff, datasets
from tqdm import tqdm
import numpy as np
import json


ROOT_PATH = Path('/storage/shafner/urban_extraction')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')
CUCD_ROOT_PATH = Path('/storage/shafner/continuous_urban_change_detection')


def aoi_ids(dataset_name: str):
    metadata_file = CUCD_ROOT_PATH / dataset_name / 'metadata.json'
    metadata = geotiff.load_json(metadata_file)
    return sorted(metadata['aois'].keys())


def produce_bua_probabilities(config_name: str, dataset_name: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading config and network
    cfg = config.load_cfg(Path.cwd() / 'configs' / f'{config_name}.yaml')
    net, _, _ = load_checkpoint(cfg.INFERENCE.CHECKPOINT, cfg, device)
    net.eval()

    dataset = datasets.CUCDDataset(cfg, dataset_name, include_geo=True)

    with torch.no_grad():
        for index in tqdm(range(len(dataset))):
            p = dataset.__getitem__(index)
            aoi_id, year, month = p['aoi_id'], p['year'], p['month']
            x = p['x'].to(device)
            logits = net(x.unsqueeze(0))
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            save_path = CUCD_ROOT_PATH / dataset_name / aoi_id / config_name
            save_path.mkdir(exist_ok=True)
            output_file = save_path / f'pred_{aoi_id}_{year}_{month:02d}.tif'
            transform, crs = p['transform'], p['crs']
            geotiff.write_tif(output_file, prob[:, :, None], transform, crs)


def produce_deep_features(config_name: str, dataset_name: str, aoi_id: str, only_first_and_last: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading config and network
    cfg = config.load_cfg(Path.cwd() / 'configs' / f'{config_name}.yaml')
    net, _, _ = load_checkpoint(cfg.INFERENCE.CHECKPOINT, cfg, device)
    net.eval()

    dataset = datasets.ContinuousUrbanChangeDetectionDataset(cfg, dataset_name, aoi_id,
                                                             only_first_and_last=only_first_and_last)

    with torch.no_grad():
        for index in tqdm(range(len(dataset))):
            p = dataset.__getitem__(index)
            aoi_id, year, month, s1, s2 = p['aoi_id'], p['year'], p['month'], p['s1'], p['s2']

            if s1 and s2:
                x_sar = p['x_s1'].to(device).unsqueeze(0)
                features_sar = net.sar_stream(x_sar)

                x_optical = p['x_s2'].to(device).unsqueeze(0)
                features_optical = net.optical_stream(x_optical)

                features_fusion = torch.cat((features_sar, features_optical), dim=1)
                features = features_fusion.squeeze().cpu().numpy().transpose((1, 2, 0))

                save_path = CUCD_ROOT_PATH / dataset_name / aoi_id / config_name
                save_path.mkdir(exist_ok=True)
                output_file = save_path / f'features_{aoi_id}_{year}_{month:02d}.npy'
                np.save(output_file, features.astype(np.float16))


def generate_metadata_file(root_path: Path, date: str, patch_size: int = 256):
    samples_file = root_path / f'samples_{date}.json'
    if samples_file.exists():
        return

    s1_path = root_path / date / 'sentinel1'
    patches = [f.stem.split('_')[-1] for f in s1_path.glob('**/*') if f.is_file()]

    max_x, max_y = 0, 0

    samples = []
    for i, patch_id in enumerate(tqdm(patches)):

        sentinel1_file = root_path / date / 'sentinel1' / f'sentinel1_stockholm_{date}_{patch_id}.tif'
        sentinel2_file = root_path / date / 'sentinel2' / f'sentinel2_stockholm_{date}_{patch_id}.tif'
        files = [sentinel1_file, sentinel2_file]

        valid = True
        for file in files:
            # if has_only_zeros(file):
            #     raise Exception(f'only zeros {file.name}')
            arr, transform, crs = geotiff.read_tif(file)
            m, n, _ = arr.shape
            if m != patch_size or n != patch_size:
                if m >= patch_size and n >= patch_size:
                    pass
                else:
                    valid = False

        sample = {
            'site': 'stockholm',
            'date': date,
            'patch_id': patch_id,
        }
        if valid:
            samples.append(sample)
            y, x = geotiff.id2yx(patch_id)
            max_x = x if x > max_x else max_x
            max_y = y if y > max_y else max_y

    # writing data to json file
    data = {
        'site': 'stockholm',
        'date': date,
        'patch_size': patch_size,
        'max_x': max_x,
        'max_y': max_y,
        'sentinel1_features': ['VV', 'VH'],
        'sentinel2_features': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
        'samples': samples
    }

    with open(str(samples_file), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def inference_stockholm_timeseries_dataset(root_path: Path, config_name: str, dates: str = None):
    dataset_path = root_path / 'stockholm_timeseries_dataset'
    assert(dataset_path.exists())

    all_dates = [d.stem for d in dataset_path.iterdir() if d.is_dir()]
    dates = all_dates if dates is None else dates

    # loading config and network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config.load_cfg(Path.cwd() / 'configs' / f'{config_name}.yaml')
    net, _, _ = load_checkpoint(cfg.INFERENCE.CHECKPOINT, cfg, device)
    net.eval()

    for date in dates:
        generate_metadata_file(dataset_path, date)
        dataset = datasets.StockholmTimeseriesDataset(cfg, dataset_path, date)

        # config inference directory
        save_path = dataset_path / date / config_name
        save_path.mkdir(exist_ok=True)

        prob_output = dataset.get_arr()
        tl_transform, tl_crs = dataset.get_geo()

        with torch.no_grad():
            for i in tqdm(range(len(dataset))):
                patch = dataset.__getitem__(i)
                img = patch['x'].to(device)
                logits = net(img.unsqueeze(0))
                prob = torch.sigmoid(logits)
                prob = prob.squeeze().cpu().numpy().astype(np.float32)
                prob = np.clip(prob, 0, 1)
                center_prob = prob[dataset.patch_size:dataset.patch_size * 2, dataset.patch_size:dataset.patch_size * 2]

                patch_id = patch['patch_id']
                transform, crs = patch['transform'], patch['crs']

                output_file = save_path / f'prob_stockholm_{date}_{patch_id}.tif'
                geotiff.write_tif(output_file, center_prob, transform, crs)

                center_prob = (center_prob * 100).astype(np.uint8)
                i_start = patch['i']
                i_end = i_start + dataset.patch_size
                j_start = patch['j']
                j_end = j_start + dataset.patch_size
                prob_output[i_start:i_end, j_start:j_end, 0] = center_prob

            save_path = dataset_path / config_name
            save_path.mkdir(exist_ok=True)
            output_file = save_path / f'prob_stockholm_{date}.tif'
            geotiff.write_tif(output_file, prob_output, tl_transform, tl_crs)


if __name__ == '__main__':
    # cfg = 'fusionda_cons05_jaccardmorelikeloss'
    cfg = 'optical_jaccardmorelikeloss'
    ds = 'spacenet7_s1s2_dataset_v3'

    produce_bua_probabilities(cfg, ds)
    # produce_deep_features(cfg, ds, aoi_id, only_first_and_last=True)
    # inference_stockholm_timeseries_dataset(Path('/storage/shafner/continuous_urban_change_detection'), cfg)

