from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from utils import geofiles, experiment_manager, networks, datasets, paths


def run_inference_spacenet7(config_name: str):

    dirs = paths.load_paths()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading config and network
    cfg = experiment_manager.load_cfg(config_name)
    net, _, _ = networks.load_checkpoint(cfg.INFERENCE.CHECKPOINT, cfg, device)
    net.eval()

    dataset = datasets.SpaceNet7Dataset(cfg)

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            patch = dataset.__getitem__(i)
            aoi_id = patch['aoi_id']
            img = patch['x'].to(device)
            logits = net(img.unsqueeze(0))
            prob = torch.sigmoid(logits) * 100
            prob = prob.squeeze().cpu().numpy().astype('uint8')
            prob = np.clip(prob, 0, 100)

            # config inference directory
            out_folder = Path(dirs.DATASET) / 'spacenet7' / config_name
            out_folder.mkdir(exist_ok=True)
            out_file = out_folder / f'{config_name}_{aoi_id}.tif'
            geofiles.write_tif(out_file, prob, patch['transform'], patch['crs'])


if __name__ == '__main__':
    run_inference_spacenet7('fusionda')
