from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from utils import geofiles, experiment_manager, networks, datasets, parsers


def preprocess_inference(cfg: experiment_manager.CfgNode, site: str):
    dataset_path = Path(cfg.PATHS.DATASET)

    # get all patches
    sentinel1_path = dataset_path / site / 'sentinel1'
    patches = ['-'.join(f.stem.split('-')[-2:]) for f in sentinel1_path.glob('**/*')]

    # renaming files (this is necessary for data downloaded from Google Drive because GEE handles file names
    # when uploading to drive instead of the cloud platform
    for sensor in ['sentinel1', 'sentinel2']:
        for patch_id in patches:
            file = dataset_path / site / sensor / f'{sensor}_{site}-{patch_id}.tif'
            new_file = dataset_path / site / sensor / f'{sensor}_{site}_{patch_id}.tif'
            if file.exists():
                file.rename(new_file)

    # create the sample file
    samples_file = dataset_path / site / 'samples.json'
    if not samples_file.exists():
        patches = [f.stem.split('_')[-1] for f in (dataset_path / site / 'sentinel1').glob('**/*')]

        data = {
            'label': 'buildings',
            'site': site,
            'patch_size': 256,
            'sentinel1_features': ['VV', 'VH'],
            'sentinel2_features': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
            'samples': []
        }

        max_x = max_y = 0
        for patch_id in patches:

            sample = {
                'site': site,
                'patch_id': patch_id,
                'is_labeled': False,
                'img_weight': -1
            }

            file = dataset_path / site / 'sentinel1' / f'sentinel1_{site}_{patch_id}.tif'
            arr, *_ = geofiles.read_tif(file)
            m, n, _ = arr.shape

            if m == 256 and n == 256:
                data['samples'].append(sample)
                y, x = int(patch_id.split('-')[0]), int(patch_id.split('-')[1])
                max_x = x if x > max_x else max_x
                max_y = y if y > max_y else max_y

        data['max_x'] = max_x
        data['max_y'] = max_y

        geofiles.write_json(samples_file, data)


def run_inference(cfg: experiment_manager.CfgNode, site: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading config and network
    net, _, _ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()

    # loading dataset from config (requires inference.json)
    dataset = datasets.TilesInferenceDataset(cfg, site)

    prob_output = dataset.get_arr()
    transform, crs = dataset.get_geo()

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            patch = dataset.__getitem__(i)
            img = patch['x'].to(device)
            logits = net(img.unsqueeze(0))
            prob = torch.sigmoid(logits) * 100
            prob = prob.squeeze().cpu().numpy().astype('uint8')
            prob = np.clip(prob, 0, 100)
            center_prob = prob[dataset.patch_size:dataset.patch_size * 2, dataset.patch_size:dataset.patch_size * 2]

            i_start = patch['i']
            i_end = i_start + dataset.patch_size
            j_start = patch['j']
            j_end = j_start + dataset.patch_size
            prob_output[i_start:i_end, j_start:j_end, 0] = center_prob

    # config inference directory
    out_folder = Path(cfg.PATHS.OUTPUT) / 'inference' / cfg.NAME
    out_folder.mkdir(exist_ok=True)
    out_file = out_folder / f'pred_{site}_{cfg.NAME}.tif'
    geofiles.write_tif(out_file, prob_output, transform, crs)


if __name__ == '__main__':
    args = parsers.inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    for site in args.sites:
        preprocess_inference(cfg, site)
        run_inference(cfg, site)
