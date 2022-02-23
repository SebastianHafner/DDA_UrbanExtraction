from pathlib import Path
from utils.metrics import *
from tqdm import tqdm
import torch
from utils.visualization import *

DATASET_PATH = Path('/storage/shafner/urban_extraction/urban_extraction_dataset')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')
NETWORK_PATH = Path('/storage/shafner/urban_extraction/networks/')
ROOT_PATH = Path('/storage/shafner/urban_extraction')


def run_quantitative_inference(config_name: str, run_type: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loading config and network
    cfg = config.load_cfg(Path.cwd() / 'configs' / f'{config_name}.yaml')
    net, _, _ = load_checkpoint(cfg.INFERENCE.CHECKPOINT, cfg, device)
    net.eval()

    # loading dataset from config (requires inference.json)
    dataset = UrbanExtractionDataset(cfg, dataset=run_type, no_augmentations=True, include_unlabeled=False)

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
        output_file = ROOT_PATH / 'validation' / f'probabilities_{run_type}_{config_name}.npy'
        output_file.parent.mkdir(exist_ok=True)
        output_data = np.stack((y_trues, y_probs))
        np.save(output_file, output_data)


def get_quantitative_data(config_name: str, run_type: str, allow_run: bool = True):
    data_file = ROOT_PATH / 'validation' / f'probabilities_{run_type}_{config_name}.npy'
    # TODO: remove
    run_quantitative_inference(config_name, run_type)
    if not data_file.exists():
        if allow_run:
            run_quantitative_inference(config_name, run_type)
        else:
            raise Exception('No data and not allowed to run quantitative inference!')
    # run_quantitative_inference(config_name, run_type)
    data = np.load(data_file, allow_pickle=True)
    return data


def plot_quantitative_validation(config_names: list, names: list, run_type: str):
    def load_data(config_name: str):
        file = DATASET_PATH.parent / 'validation' / f'validation_{config_name}.json'
        d = load_json(file)
        return d[run_type]

    data = [load_data(config_name) for config_name in config_names]
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


def show_quantitative_results(config_name: str, run_type: str):
    data = get_quantitative_data(config_name, run_type)
    y_true = data[0, ]
    y_prob = data[1, ]
    f1 = f1_score_from_prob(y_prob, y_true, 0.5)
    prec = precsision_from_prob(y_prob, y_true, 0.5)
    rec = recall_from_prob(y_prob, y_true, 0.5)
    print(f'total: {f1:.3f} f1 score, {prec:.3f} precision, {rec:.3f} recall')



if __name__ == '__main__':
    config_name = 'optical'

    config_names = ['sar_jaccardmorelikeloss', 'optical_jaccardmorelikeloss', 'fusion_jaccardmorelikeloss',
                    'fusionda_cons05_jaccardmorelikeloss']
    names = ['SAR', 'Optical', 'Fusion', 'Fusion-DA']
    # plot_threshold_dependency(config_names, 'training', names)
    # for config_name in config_names:
    #     show_quantitative_results(config_name, 'validation')
    show_quantitative_results('igarss_sar', 'validation')