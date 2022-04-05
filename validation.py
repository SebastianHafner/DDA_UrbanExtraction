import torch
from utils import networks, experiment_manager, datasets, parsers, metrics, geofiles
from tqdm import tqdm
from pathlib import Path
import numpy as np


def run_quantitative_inference(cfg: experiment_manager.CfgNode, run_type: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _, _ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()

    dataset = datasets.UrbanExtractionDataset(cfg, dataset=run_type, no_augmentations=True, include_unlabeled=False)

    y_probs, y_trues = [], []

    with torch.no_grad():
        for index in tqdm(range(len(dataset))):
            sample = dataset.__getitem__(index)
            img = sample['x'].to(device)
            y_prob = net(img.unsqueeze(0))
            y_prob = torch.sigmoid(y_prob).flatten().cpu().numpy()
            y_true = sample['y'].flatten().cpu().numpy()
            y_probs.append(y_prob)
            y_trues.append(y_true)

        y_probs, y_trues = np.concatenate(y_probs, axis=0), np.concatenate(y_trues, axis=0)
        output_file = Path(cfg.PATHS.OUTPUT) / 'validation' / f'probabilities_{run_type}_{cfg.NAME}.npy'
        output_file.parent.mkdir(exist_ok=True)
        output_data = np.stack((y_trues, y_probs))
        np.save(output_file, output_data)


def get_quantitative_data(cfg: experiment_manager.CfgNode, run_type: str):
    data_file = Path(cfg.PATHS.OUTPUT) / 'validation' / f'probabilities_{run_type}_{cfg.NAME}.npy'
    if not data_file.exists():
        run_quantitative_inference(cfg, run_type)
    data = np.load(data_file, allow_pickle=True)
    return data


def get_quantitative_results(cfg: experiment_manager.CfgNode, run_type: str):
    data = get_quantitative_data(cfg, run_type)
    y_true = data[0,]
    y_prob = data[1,]
    data = {
        'f1_score': metrics.f1_score_from_prob(y_prob, y_true),
        'precision': metrics.precision_from_prob(y_prob, y_true),
        'recall': metrics.recall_from_prob(y_prob, y_true),
        'iou': metrics.iou_from_prob(y_prob, y_true),
        'kappa': metrics.kappa_from_prob(y_prob, y_true),
    }
    data_file = Path(cfg.PATHS.OUTPUT) / 'validation' / f'quantitative_results_{run_type}_{cfg.NAME}.json'
    geofiles.write_json(data_file, data)


if __name__ == '__main__':
    args = parsers.testing_inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    get_quantitative_results(cfg, 'validation')

