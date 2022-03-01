from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from utils import metrics, geofiles, experiment_manager, networks, datasets, visualization

FONTSIZE = 18
mpl.rcParams.update({'font.size': FONTSIZE})


def spacenet7_aoi_ids() -> list:
    dirs = paths.load_paths()
    file = Path(dirs.DATASET) / 'spacenet7' / 'samples.json'
    metadata = geofiles.load_json(file)
    aoi_ids = [s['aoi_id'] for s in metadata['samples']]
    return sorted(aoi_ids)


def run_quantitative_evaluation(config_name: str, site: str, threshold: float = None, save_output: bool = False):
    print(f'running quantitative evaluation for {site} with {config_name}...')

    dirs = paths.load_paths()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading config and network
    cfg = experiment_manager.load_cfg(config_name)
    net, _, _ = networks.load_checkpoint(cfg.INFERENCE.CHECKPOINT, cfg, device)
    net.eval()

    # loading dataset from config (requires inference.json)
    dataset = datasets.TilesInferenceDataset(cfg, site)

    y_probs, y_trues = None, None

    thresh = threshold if threshold else cfg.INFERENCE.THRESHOLDS.VALIDATION

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
            output_path = Path(dirs.OUTPUT) / 'quantitative_evaluation' / config_name
            output_path.mkdir(exist_ok=True)
            output_file = output_path / f'{site}_{config_name}.npy'
            np.save(output_file, output_data)
        else:
            y_preds = (y_probs > thresh).float()
            prec = metrics.precision(y_trues, y_preds, dim=0)
            rec = metrics.recall(y_trues, y_preds, dim=0)
            f1 = metrics.f1_score(y_trues, y_preds, dim=0)
            print(f'{site}: f1 score {f1:.3f} - precision {prec:.3f} - recall {rec:.3f}')


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


def run_quantitative_inference_sota(sota_name: str, threshold: float):
    dirs = paths.load_paths()
    cfg = experiment_manager.load_cfg('base')
    dataset = datasets.SpaceNet7Dataset(cfg)

    data = {}
    for index in tqdm(range(len(dataset))):
        test_site = dataset.__getitem__(index)
        aoi_id = test_site['aoi_id']

        sota, *_ = geofiles.read_tif(Path(dirs.DATASET) / 'spacenet7' / sota_name / f'{sota_name}_{aoi_id}.tif')
        sota = sota.flatten().astype(np.float32)
        if sota_name == 'wsf2019':
            sota = sota / 255
        y_true = test_site['y'].flatten().cpu().numpy()

        group_name = test_site['group_name']
        if group_name not in data.keys():
            data[group_name] = []

        f1 = metrics.f1_score_from_prob(sota, y_true, threshold)
        p = metrics.precsision_from_prob(sota, y_true, threshold)
        r = metrics.recall_from_prob(sota, y_true, threshold)

        site_data = {
            'aoi_id': test_site['aoi_id'],
            'y_prob': sota,
            'y_true': y_true,
            'f1_score': f1,
            'precision': p,
            'recall': r,
        }

        data[group_name].append(site_data)

        output_file = Path(dirs.OUTPUT) / 'testing' / f'probabilities_{sota_name}.npy'
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


def qualitative_testing(aoi_id: str, config: str, save_plot: bool = False):
    dirs = paths.load_paths()

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    for _, ax in np.ndenumerate(axs):
        ax.set_xticks([])
        ax.set_yticks([])

    ax_sar = axs[0, 0]
    sar_file = Path(dirs.DATASET) / 'spacenet7' / 'sentinel1' / f'sentinel1_{aoi_id}.tif'
    visualization.plot_sar(ax_sar, sar_file)
    ax_sar.set_xlabel(f'(a) SAR (VV)', fontsize=FONTSIZE)
    ax_sar.xaxis.set_label_coords(0.5, -0.025)

    ax_opt = axs[0, 1]
    opt_file = Path(dirs.DATASET) / 'spacenet7' / 'sentinel2' / f'sentinel2_{aoi_id}.tif'
    visualization.plot_optical(ax_opt, opt_file)
    ax_opt.set_xlabel(f'(b) Optical (True Color)', fontsize=FONTSIZE)
    ax_opt.xaxis.set_label_coords(0.5, -0.025)

    ax_sn7 = axs[0, 2]
    sn7_file = Path(dirs.DATASET) / 'spacenet7' / 'buildings' / f'buildings_{aoi_id}.tif'
    visualization.plot_buildings(ax_sn7, sn7_file, 0)
    ax_sn7.set_xlabel(f'(c) SpaceNet7 Ground Truth', fontsize=FONTSIZE)
    ax_sn7.xaxis.set_label_coords(0.5, -0.025)

    ax_ghs = axs[1, 0]
    ghs_file = Path(dirs.DATASET) / 'spacenet7' / 'ghs' / f'ghs_{aoi_id}.tif'
    visualization.plot_buildings(ax_ghs, ghs_file, 0.2)
    ax_ghs.set_xlabel(f'(d) GHS-S2', fontsize=FONTSIZE)
    ax_ghs.xaxis.set_label_coords(0.5, -0.025)

    ax_wsf2019 = axs[1, 1]
    wsf2019_file = Path(dirs.DATASET) / 'spacenet7' / 'wsf2019' / f'wsf2019_{aoi_id}.tif'
    visualization.plot_buildings(ax_wsf2019, wsf2019_file, 0)
    ax_wsf2019.set_xlabel(f'(e) WSF2019', fontsize=FONTSIZE)
    ax_wsf2019.xaxis.set_label_coords(0.5, -0.025)

    ax_ours = axs[1, 2]
    ours_file = Path(dirs.DATASET) / 'spacenet7' / config / f'{config}_{aoi_id}.tif'
    visualization.plot_buildings(ax_ours, ours_file, 50)
    ax_ours.set_xlabel(f'(f) Ours (Fusion-DA)', fontsize=FONTSIZE)
    ax_ours.xaxis.set_label_coords(0.5, -0.025)

    plt.tight_layout()

    if save_plot:
        plot_file = Path(dirs.OUTPUT) / 'plots' / 'qualitative_comparison' / f'qualitative_comparison_{aoi_id}.png'
        plot_file.parent.mkdir(exist_ok=True)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def histogram_testing(config_names: list, names: list, save_plot: bool = False):
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    for i, group_name in enumerate(GROUP_NAMES):
        ax_i = i // 3
        ax_j = i % 3
        ax = axs[ax_i, ax_j]
        for j, config_name in enumerate(config_names):
            data = get_quantitative_data_sn7(config_name)
            y_prob = np.concatenate([site['y_prob'] for site in data[group_name]]).flatten()
            weights = np.ones_like(y_prob) / len(y_prob)
            ax.hist(y_prob, weights=weights, bins=25, alpha=0.6, label=names[j])
        if ax_j == 0:
            ax.set_ylabel('Frequency (%)', fontsize=FONTSIZE)
        if ax_i == 1:
            ax.set_xlabel('CNN output probability', fontsize=FONTSIZE)
        group_name = 'Asia' if group_name == 'AS' else group_name
        ax.text(0.1, 0.087, group_name, fontsize=24)
        ax.yaxis.grid(True)

        xticks = np.linspace(0, 1, 5)
        xticklabels = [f'{tick:.1f}' for tick in xticks]
        xticklabels[0] = '0'
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=FONTSIZE)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 0.1))
        yticks = np.linspace(0, 0.1, 6)
        ax.set_yticks(yticks)
        yticklabels = [f'{tick * 100:.0f}' for tick in yticks]
        ax.set_yticklabels(yticklabels, fontsize=FONTSIZE)

    # handles = [Patch(facecolor=COLORS[i], edgecolor=COLORS[i]) for i in range(g)]
    # axs[1, 0].legend(handles, names, loc='lower left', ncol=2, frameon=False, handletextpad=0.5,
    #                  columnspacing=0.8, handlelength=0.6, fontsize=FONTSIZE)
    plt.tight_layout()
    plt.legend(frameon=False, handletextpad=0.5, columnspacing=0.8, handlelength=0.6)
    if save_plot:
        dirs = paths.load_paths()
        output_file = Path(dirs.OUTPUT) / 'plots' / f'test_histograms.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


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


def plot_consistency_impact(config_names: list, save_plot: bool = True):
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    accuracy_metrics = ['F1 score', 'Precision', 'Recall']
    for i, group_name in enumerate(GROUP_NAMES):
        ax_i = i // 3
        ax_j = i % 3
        ax = axs[ax_i, ax_j]
        cons_values, accuracy_values = [], [[], [], []]
        for j, config_name in enumerate(config_names):
            data = get_quantitative_data_sn7(config_name)
            y_prob = np.concatenate([site['y_prob'] for site in data[group_name]]).flatten()
            y_true = np.concatenate([site['y_true'] for site in data[group_name]]).flatten()
            accuracy_values[0].append(metrics.f1_score_from_prob(y_prob, y_true, 0.5))
            accuracy_values[1].append(metrics.precsision_from_prob(y_prob, y_true, 0.5))
            accuracy_values[2].append(metrics.recall_from_prob(y_prob, y_true, 0.5))
            cfg = experiment_manager.load_cfg(config_name)
            cons_values.append(cfg.CONSISTENCY_TRAINER.LOSS_FACTOR)
        for i, metric in enumerate(accuracy_metrics):
            if metric == 'Precision' or metric == 'Recall':
                ax.plot(cons_values, accuracy_values[i], marker='o', ms=8, lw=2, label=metric)
        group_name = 'Asia' if group_name == 'AS' else group_name
        ax.text(0.9, 0.9, group_name, fontsize=24, horizontalalignment='right', verticalalignment='top')
        if ax_j == 0:
            ax.set_ylabel('Accuracy', fontsize=FONTSIZE)
        if ax_i == 1:
            ax.set_xlabel(r'$\varphi$ (consistency impact)', fontsize=FONTSIZE)
    for _, ax in np.ndenumerate(axs):
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
    axs[0, 0].legend(frameon=False, fontsize=FONTSIZE)
    if save_plot:
        dirs = paths.load_paths()
        output_file = Path(dirs.OUTPUT) / 'plots' / f'test_consistency_impact.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':


    config_names = ['sar', 'optical', 'fusion', 'fusionda_spacenet7', 'ghs', 'wsf2019']
    names = ['SAR', 'Opt', 'Fus.', 'Fus.-DA', 'GHS', 'WSF']



    plot_boxplots_spacenet7('f1_score', 'F1 score', config_names, names, 4, save_plot=True)
    # histogram_testing(['ghs', 'fusionda_spacenet7'], ['GHS-S2', 'Fusion-DA'], save_plot=True)
    # plot_consistency_impact(['fusionda_spacenet7_cons0', 'fusionda_spacenet7_cons025', 'fusionda_spacenet7',
    #                          'fusionda_spacenet7_cons075', 'fusionda_spacenet7_cons1'])
    # plot_activation_comparison_sn7(config_names, save_plots=False)
    # plot_activation_comparison_assembled_sn7(config_names, names, aoi_ids, save_plot=False)

    # plot_precision_recall_curve_sn7(config_names, names, save_plot=False)
    # plot_threshold_dependency(config_names, names)

if __name__ == '__main__':
    pass