from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from utils import metrics, geofiles, experiment_manager, networks, datasets, paths, visualization
from sklearn.metrics import precision_recall_curve
import pandas as pd

FONTSIZE = 18
mpl.rcParams.update({'font.size': FONTSIZE})


def spacenet7_aoi_ids() -> list:
    dirs = paths.load_paths()
    file = Path(dirs.DATASET) / 'spacenet7' / 'samples.json'
    metadata = geofiles.load_json(file)
    aoi_ids = [s['aoi_id'] for s in metadata['samples']]
    return sorted(aoi_ids)


# SpaceNet7 testing

GROUPS = [(1, 'NA_AU', '#63cd93'), (2, 'SA', '#f0828f'), (3, 'EU', '#6faec9'), (4, 'SSA', '#5f4ad9'),
          (5, 'NAF_ME', '#8dee47'), (6, 'AS', '#d9b657'), ('total', 'Total', '#ffffff')]
GROUP_NAMES = ['NA_AU', 'SA', 'EU', 'SSA', 'NAF_ME', 'AS']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2']


def run_quantitative_inference_sn7(config_name: str, threshold: float = 0.5):
    # loading config and network
    cfg = experiment_manager.load_cfg(config_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _, _ = networks.load_checkpoint(cfg.INFERENCE.CHECKPOINT, cfg, device)
    net.eval()

    # loading dataset from config (requires inference.json)
    dataset = datasets.SpaceNet7Dataset(cfg)

    data = {}
    with torch.no_grad():
        for index in tqdm(range(len(dataset))):
            test_site = dataset.__getitem__(index)
            aoi_id = test_site['aoi_id']
            # if aoi_id == 'L15-0434E-1218N_1736_3318_13':
            #     continue
            img = test_site['x'].to(device)
            y_prob = net(img.unsqueeze(0))
            y_prob = torch.sigmoid(y_prob).flatten().cpu().numpy()
            y_true = test_site['y'].flatten().cpu().numpy()

            group_name = test_site['group_name']
            if group_name not in data.keys():
                data[group_name] = []

            f1 = metrics.f1_score_from_prob(y_prob, y_true, threshold)
            p = metrics.precsision_from_prob(y_prob, y_true, threshold)
            r = metrics.recall_from_prob(y_prob, y_true, threshold)

            site_data = {
                'aoi_id': test_site['aoi_id'],
                'y_prob': y_prob,
                'y_true': y_true,
                'f1_score': f1,
                'precision': p,
                'recall': r,
            }

            data[group_name].append(site_data)

        dirs = paths.load_paths()
        output_file = Path(dirs.OUTPUT) / 'testing' / f'probabilities_{config_name}.npy'
        output_file.parent.mkdir(exist_ok=True)
        np.save(output_file, data)




def get_quantitative_data_sn7(config_name: str, allow_run: bool = True):
    dirs = paths.load_paths()
    data_file = Path(dirs.OUTPUT) / 'testing' / f'probabilities_{config_name}.npy'
    if not data_file.exists():
        if allow_run:
            if config_name == 'wsf2019' or config_name == 'ghs':
                run_quantitative_inference_sota(config_name, threshold=0.2)
            else:
                run_quantitative_inference_sn7(config_name, threshold=0.5)
        else:
            raise Exception('No data and not allowed to run quantitative inference!')
    data = np.load(data_file, allow_pickle=True)
    data = dict(data[()])
    return data


def show_quantitative_testing_sn7(config_name: str, threshold: float):
    print(f'{"-" * 10} {config_name} {"-" * 10}')

    data = get_quantitative_data_sn7(config_name)
    for metric in ['f1_score', 'precision', 'recall']:
        print(metric)
        region_values = []
        for region_name, region in data.items():

            y_true = np.concatenate([site['y_true'] for site in region], axis=0)
            y_prob = np.concatenate([site['y_prob'] for site in region], axis=0)

            if metric == 'f1_score':
                value = metrics.f1_score_from_prob(y_prob, y_true, threshold)
            elif metric == 'precision':
                value = metrics.precsision_from_prob(y_prob, y_true, threshold)
            else:
                value = metrics.recall_from_prob(y_prob, y_true, threshold)

            print(f'{region_name}: {value:.3f},', end=' ')
            region_values.append(value)

        print('')
        min_ = np.min(region_values)
        max_ = np.max(region_values)
        mean = np.mean(region_values)
        std = np.std(region_values)

        print(f'summary statistics: {min_:.3f} min, {max_:.3f} max, {mean:.3f} mean, {std:.3f} std')

    y_true = np.concatenate([site['y_true'] for region in data.values() for site in region], axis=0)
    y_prob = np.concatenate([site['y_prob'] for region in data.values() for site in region], axis=0)
    f1 = metrics.f1_score_from_prob(y_prob, y_true, 0.5)
    prec = metrics.precsision_from_prob(y_prob, y_true, 0.5)
    rec = metrics.recall_from_prob(y_prob, y_true, 0.5)
    print(f'total: {f1:.3f} f1 score, {prec:.3f} precision, {rec:.3f} recall')



# https://matplotlib.org/stable/gallery/statistics/boxplot_color.html
def plot_boxplots_spacenet7(metric: str, metric_name: str, config_names: list, names: list, gap_index: int,
                            save_plot: bool = False):
    box_width = 0.4

    def set_box_color(bp, color):
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    fig, axs = plt.subplots(2, 3, figsize=(16, 12))
    for i, group_name in enumerate(GROUP_NAMES):
        ax_i = i // 3
        ax_j = i % 3
        ax = axs[ax_i, ax_j]
        boxplot_data = []
        for j, config_name in enumerate(config_names):
            data = get_quantitative_data_sn7(config_name)
            group_data = [site[metric] for site in data[group_name]]
            #  if not site['aoi_id'] == 'L15-0434E-1218N_1736_3318_13'
            boxplot_data.append(group_data)

        x_positions = np.arange(len(config_names))
        for index in range(gap_index, len(config_names)):
            # x_positions[index] = x_positions[index] + 1
            pass
        bplot = ax.boxplot(
            boxplot_data,
            positions=x_positions,
            patch_artist=True,
            widths=box_width,
            whis=[0, 100],
            medianprops={"linewidth": 1, "solid_capstyle": "butt"},
        )
        ax.plot([gap_index - 0.5, gap_index - 0.5], [0, 1], '--', c='k', )
        for i_patch, patch in enumerate(bplot['boxes']):
            if i_patch < gap_index:
                patch.set_facecolor(COLORS[i_patch])
            else:
                patch.set_facecolor('white')
        set_box_color(bplot, 'k')
        ax.set_xlim((-0.5, len(config_names) - 0.5))
        ax.set_ylim((0, 1))
        if ax_j == 0:
            ax.set_ylabel(metric_name, fontsize=FONTSIZE)
        group_name = 'Asia' if group_name == 'AS' else group_name
        ax.text(-0.2, 0.87, group_name, fontsize=24)
        ax.yaxis.grid(True)

        x_ticks = [(gap_index - 1) / 2] + [i for i in range(gap_index, len(config_names))]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(['Ours'] + [names[i] for i in range(gap_index, len(config_names))], fontsize=FONTSIZE)

    handles = [Patch(facecolor=COLORS[i], edgecolor=COLORS[i]) for i in range(gap_index)]
    axs[1, 0].legend(handles, names, loc='lower left', ncol=2, frameon=False, handletextpad=0.5,
                     columnspacing=0.8, handlelength=0.6, fontsize=FONTSIZE)
    plt.tight_layout()
    if save_plot:
        dirs = paths.load_paths()
        output_file = Path(dirs.OUTPUT) / 'plots' / f'test_f1_scores.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def produce_data_file(config_name: str, threshold: float = 0.5):
    dirs = paths.load_paths()
    csv_data = {'aoi_id': [], 'f1_score': [], 'precision': [], 'recall': [], 'region': []}
    data = get_quantitative_data_sn7(config_name)
    for region, regional_data in data.items():
        for aoi_data in regional_data:
            csv_data['aoi_id'].append(aoi_data['aoi_id'])
            csv_data['region'].append(region)
            for metric in ['f1_score', 'precision', 'recall']:
                y_true = aoi_data['y_true']
                y_prob = aoi_data['y_prob']
                if metric == 'f1_score':
                    value = metrics.f1_score_from_prob(y_prob, y_true, threshold)
                elif metric == 'precision':
                    value = metrics.precsision_from_prob(y_prob, y_true, threshold)
                else:
                    value = metrics.recall_from_prob(y_prob, y_true, threshold)
                csv_data[metric].append(value)
    df = pd.DataFrame.from_dict(csv_data)
    file = Path(dirs.OUTPUT) / f'geovisualization_project_data.csv'
    df.to_csv(file, sep=';')


def produce_plots(config_name: str):
    dirs = paths.load_paths()

    aoi_ids = spacenet7_aoi_ids()
    for aoi_id in aoi_ids:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        for _, ax in np.ndenumerate(axs):
            ax.set_xticks([])
            ax.set_yticks([])

        # sentinel-1
        ax_sar = axs[0, 0]
        sar_file = Path(dirs.DATASET) / 'spacenet7' / 'sentinel1' / f'sentinel1_{aoi_id}.tif'
        visualization.plot_sar(ax_sar, sar_file)
        ax_sar.set_xlabel(f'(a) SAR (VV)', fontsize=FONTSIZE)
        ax_sar.xaxis.set_label_coords(0.5, -0.025)

        # sentinel-2
        ax_opt = axs[0, 1]
        opt_file = Path(dirs.DATASET) / 'spacenet7' / 'sentinel2' / f'sentinel2_{aoi_id}.tif'
        visualization.plot_optical(ax_opt, opt_file)
        ax_opt.set_xlabel(f'(b) Optical (True Color)', fontsize=FONTSIZE)
        ax_opt.xaxis.set_label_coords(0.5, -0.025)

        ax_sn7 = axs[1, 0]
        sn7_file = Path(dirs.DATASET) / 'spacenet7' / 'buildings' / f'buildings_{aoi_id}.tif'
        visualization.plot_buildings(ax_sn7, sn7_file, 0)
        ax_sn7.set_xlabel(f'(c) SpaceNet7 Ground Truth', fontsize=FONTSIZE)
        ax_sn7.xaxis.set_label_coords(0.5, -0.025)

        ax_ours = axs[1, 1]
        ours_file = Path(dirs.DATASET) / 'spacenet7' / config / f'{config}_{aoi_id}.tif'
        visualization.plot_buildings(ax_ours, ours_file, 50)
        ax_ours.set_xlabel(f'(f) Ours (Fusion-DA)', fontsize=FONTSIZE)
        ax_ours.xaxis.set_label_coords(0.5, -0.025)

        plt.tight_layout()
        plot_file = Path(dirs.OUTPUT) / 'plots' / 'geovisualization' / f'{aoi_id}.jpg'
        plot_file.parent.mkdir(exist_ok=True)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    config = 'fusionda_spacenet7'
    # produce_plots(config)
    produce_data_file(config)
