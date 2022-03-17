from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from utils import geofiles, experiment_manager, networks, datasets, parsers


def run_inference(config_name: str, site: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading config and network
    cfg = experiment_manager.load_cfg(config_name)
    net, _, _ = networks.load_checkpoint(cfg.INFERENCE.CHECKPOINT, cfg, device)
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
    out_folder = Path(cfg.PATHS.OUTPUT) / 'inference' / config_name
    out_folder.mkdir(exist_ok=True)
    out_file = out_folder / f'buap_{site}_{config_name}.tif'
    geofiles.write_tif(out_file, prob_output, transform, crs)


def produce_label(config_name: str, site: str):
    print(f'producing label for {site} with {config_name}...')

    dirs = paths.load_paths()

    # loading config and dataset
    cfg = experiment_manager.load_cfg(config_name)
    dataset = datasets.TilesInferenceDataset(cfg, site)

    label_output = dataset.get_arr()
    transform, crs = dataset.get_geo()

    for i in tqdm(range(len(dataset))):
        patch = dataset.__getitem__(i)
        label = patch['y'].cpu().numpy().squeeze().astype('uint8')
        label = np.clip(label * 100, 0, 100)

        i_start = patch['i']
        i_end = i_start + dataset.patch_size
        j_start = patch['j']
        j_end = j_start + dataset.patch_size
        label_output[i_start:i_end, j_start:j_end, 0] = label

    save_path = Path(dirs.OUTPUT) / 'inference' / config_name
    save_path.mkdir(exist_ok=True)
    output_file = save_path / f'label_{site}_{config_name}.tif'
    geofiles.write_tif(output_file, label_output, transform, crs)


if __name__ == '__main__':
    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)


    run_inference_spacenet7('fusionda')
