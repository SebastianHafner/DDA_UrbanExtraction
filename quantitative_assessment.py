from pathlib import Path
from networks.network_loader import load_network, load_checkpoint
from utils.datasets import TilesInferenceDataset
from experiment_manager.config import config
from utils.metrics import *
from tqdm import tqdm
import torch
import numpy as np
import json
import matplotlib.pyplot as plt

DATASET_PATH = Path('/storage/shafner/urban_extraction/urban_extraction')
NETWORK_PATH = Path('/storage/shafner/urban_extraction/networks/')
ROOT_PATH = Path('/storage/shafner/urban_extraction')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')


def compute_accuracy_metrics(config_name: str, checkpoint: int, output_file: Path = None):

    cfg_file = CONFIG_PATH / f'{config_name}.yaml'
    cfg = config.load_cfg(cfg_file)

    # loading network
    net_file = NETWORK_PATH / f'{config_name}_{checkpoint}.pkl'
    net = load_network(cfg, net_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    data = {'training': {}, 'validation': {}}
    for run_type in ['training', 'validation']:
        sites = cfg.DATASETS.SITES.TRAINING if run_type == 'training' else cfg.DATASETS.SITES.VALIDATION
        for site in sites:
            print(f'Quantitative assessment {site} ({run_type})')
            dataset = UrbanExtractionDataset(cfg=cfg, dataset=site, no_augmentations=True)
            y_true_set = np.array([])
            y_pred_set = np.array([])

            for index in tqdm(range(len(dataset))):
                sample = dataset.__getitem__(index)

                with torch.no_grad():
                    x = sample['x'].to(device)
                    y_true = sample['y'].to(device)
                    logits = net(x.unsqueeze(0))
                    y_pred = torch.sigmoid(logits) > cfg.THRESH

                    y_true = y_true.detach().cpu().flatten().numpy()
                    y_pred = y_pred.detach().cpu().flatten().numpy()
                    y_true_set = np.concatenate((y_true_set, y_true))
                    y_pred_set = np.concatenate((y_pred_set, y_pred))

            y_true_set, y_pred_set = torch.Tensor(np.array(y_true_set)), torch.Tensor(np.array(y_pred_set))
            prec = precision(y_true_set, y_pred_set, dim=0)
            rec = recall(y_true_set, y_pred_set, dim=0)
            f1 = f1_score(y_true_set, y_pred_set, dim=0)

            print(f'Precision: {prec.item():.3f} - Recall: {rec.item():.3f} - F1 score: {f1.item():.3f}')

            data[run_type][site] = {'f1_score': f1.item(), 'precision': prec.item(), 'recall': rec.item()}

    if output_file is not None:
        with open(str(output_file), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def plot_quantitative_results(files: list, names: list, run_type: str):

    def load_data(file: Path):
        with open(str(file)) as f:
            d = json.load(f)
        return d[run_type]

    data = [load_data(file) for file in files]
    width = 0.2

    metrics = ['f1_score', 'precision', 'recall']
    for i, metric in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(10, 4))
        for j, experiment in enumerate(data):
            sites = experiment.keys()
            ind = np.arange(len(sites))
            experiment_data = [experiment[site][metric] for site in sites]
            ax.bar(ind + (j * width), experiment_data, width, label=names[j], zorder=3)

        ax.set_ylim((0, 1))
        ax.set_ylabel(metric)
        ax.legend(loc='best')
        ax.set_xticks(ind)
        ax.set_xticklabels(sites)
        plt.grid(b=True, which='major', axis='y', zorder=0)
        plt.show()


def run_quantitative_assessment(config_name: str, save_output: bool = False):

    # loading config and network
    cfg = config.load_cfg(Path.cwd() / 'configs' / f'{config_name}.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _, _ = load_checkpoint(cfg.INFERENCE.CHECKPOINT, cfg, device)
    net.eval()

    sites = {
        'validation': cfg.DATASETS.VALIDATION,
        'test': cfg.DATASETS.UNLABELED
    }

    for dataset_name, sites in sites.items():
        y_probs, y_trues = None, None
        for site in sites:
            dataset = TilesInferenceDataset(cfg, site)
            with torch.no_grad():
                for i in tqdm(range(len(dataset))):
                    patch = dataset.__getitem__(i)
                    img = patch['x'].to(device)
                    logits = net(img.unsqueeze(0))
                    prob = torch.sigmoid(logits).squeeze()

                    center_prob = prob[dataset.patch_size:dataset.patch_size * 2, dataset.patch_size:dataset.patch_size * 2]
                    center_prob = center_prob.flatten().float().cpu()

                    assert (patch['is_labeled'])
                    label = patch['y'].flatten().float().cpu()

                    if y_probs is not None:
                        y_probs = torch.cat((y_probs, center_prob), dim=0)
                        y_trues = torch.cat((y_trues, label), dim=0)
                    else:
                        y_probs = center_prob
                        y_trues = label

        if save_output:
            y_probs = y_probs.numpy()
            y_trues = y_trues.numpy()
            output_data = np.stack((y_trues, y_probs))
            output_path = ROOT_PATH / 'quantitative_evaluation' / config_name
            output_path.mkdir(exist_ok=True)

            output_file = output_path / f'{"".join(sites)}_{config_name}.npy'
            np.save(output_file, output_data)
        else:
            y_preds = (y_probs > 0.5).float()
            prec = precision(y_trues, y_preds, dim=0)
            rec = recall(y_trues, y_preds, dim=0)
            f1 = f1_score(y_trues, y_preds, dim=0)
            print(f'{dataset_name}: f1 score {f1:.3f} - precision {prec:.3f} - recall {rec:.3f}')


if __name__ == '__main__':
    config_name = 'igarss_fusion'
    run_quantitative_assessment(config_name)
