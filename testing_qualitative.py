from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from utils import geofiles, experiment_manager, visualization, parsers

FONTSIZE = 18
mpl.rcParams.update({'font.size': FONTSIZE})


def get_spacenet7_aoi_ids(dataset_path: str) -> list:
    file = Path(dataset_path) / 'spacenet7' / 'spacenet7_regions.json'
    metadata_regions = geofiles.load_json(file)
    aoi_ids = metadata_regions['data'].keys()
    return sorted(aoi_ids)


def get_region_names(dataset_path: str) -> list:
    file = Path(dataset_path) / 'spacenet7' / 'spacenet7_regions.json'
    metadata_regions = geofiles.load_json(file)
    n_regions = len(metadata_regions['regions'].keys())
    region_names = [metadata_regions['regions'][str(i)] for i in range(n_regions)]
    return region_names


def get_ghs_threshold(dataset_path: str, aoi_id: str) -> float:
    file = Path(dataset_path) / 'spacenet7' / 'ghs_thresholds.json'
    ghs_thresholds = geofiles.load_json(file)
    threshold = float(ghs_thresholds[aoi_id])
    return threshold


def get_quantitative_data(output_path: str, config_name: str):
    data_file = Path(output_path) / 'testing' / f'probabilities_{config_name}.npy'
    assert(data_file.exists())
    data = np.load(data_file, allow_pickle=True)
    data = dict(data[()])
    return data


def qualitative_sota_comparison(cfg: experiment_manager.CfgNode):
    dataset_path = cfg.PATHS.DATASET
    aoi_ids = get_spacenet7_aoi_ids(dataset_path)
    for aoi_id in aoi_ids:
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        for _, ax in np.ndenumerate(axs):
            ax.set_xticks([])
            ax.set_yticks([])

        ax_sar = axs[0, 0]
        sar_file = Path(dataset_path) / 'spacenet7' / 'sentinel1' / f'sentinel1_{aoi_id}.tif'
        visualization.plot_sar(ax_sar, sar_file)
        ax_sar.set_xlabel(f'(a) SAR (VV)', fontsize=FONTSIZE)
        ax_sar.xaxis.set_label_coords(0.5, -0.025)

        ax_opt = axs[0, 1]
        opt_file = Path(dataset_path) / 'spacenet7' / 'sentinel2' / f'sentinel2_{aoi_id}.tif'
        visualization.plot_optical(ax_opt, opt_file)
        ax_opt.set_xlabel(f'(b) Optical (True Color)', fontsize=FONTSIZE)
        ax_opt.xaxis.set_label_coords(0.5, -0.025)

        ax_sn7 = axs[0, 2]
        sn7_file = Path(dataset_path) / 'spacenet7' / 'buildings' / f'buildings_{aoi_id}.tif'
        visualization.plot_buildings(ax_sn7, sn7_file, 0)
        ax_sn7.set_xlabel(f'(c) SpaceNet7 Ground Truth', fontsize=FONTSIZE)
        ax_sn7.xaxis.set_label_coords(0.5, -0.025)

        ax_ghs = axs[1, 0]
        ghs_file = Path(dataset_path) / 'spacenet7' / 'ghs' / f'ghs_{aoi_id}.tif'
        visualization.plot_buildings(ax_ghs, ghs_file, get_ghs_threshold(dataset_path, aoi_id))
        ax_ghs.set_xlabel(f'(d) GHS-S2', fontsize=FONTSIZE)
        ax_ghs.xaxis.set_label_coords(0.5, -0.025)

        ax_wsf2019 = axs[1, 1]
        wsf2019_file = Path(dataset_path) / 'spacenet7' / 'wsf2019' / f'wsf2019_{aoi_id}.tif'
        visualization.plot_buildings(ax_wsf2019, wsf2019_file, 0)
        ax_wsf2019.set_xlabel(f'(e) WSF2019', fontsize=FONTSIZE)
        ax_wsf2019.xaxis.set_label_coords(0.5, -0.025)

        ax_ours = axs[1, 2]
        ours_file = Path(dataset_path) / 'spacenet7' / cfg.NAME / f'{cfg.NAME}_{aoi_id}.tif'
        visualization.plot_buildings(ax_ours, ours_file, 0.5)
        ax_ours.set_xlabel(f'(f) Ours (Fusion-DA)', fontsize=FONTSIZE)
        ax_ours.xaxis.set_label_coords(0.5, -0.025)

        plt.tight_layout()

        plot_file = Path(cfg.PATHS.OUTPUT) / 'plots' / 'qualitative_comparison' / f'qualitative_comparison_{aoi_id}.png'
        plot_file.parent.mkdir(exist_ok=True)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)


def qualitative_results(cfg: experiment_manager.CfgNode):
    dataset_path = cfg.PATHS.DATASET
    aoi_ids = get_spacenet7_aoi_ids(dataset_path)
    for aoi_id in aoi_ids:
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        for _, ax in np.ndenumerate(axs):
            ax.set_xticks([])
            ax.set_yticks([])

        ax_sar = axs[0]
        sar_file = Path(dataset_path) / 'spacenet7' / 'sentinel1' / f'sentinel1_{aoi_id}.tif'
        visualization.plot_sar(ax_sar, sar_file)
        ax_sar.set_xlabel(f'(a) SAR (VV)', fontsize=FONTSIZE)
        ax_sar.xaxis.set_label_coords(0.5, -0.025)

        ax_opt = axs[1]
        opt_file = Path(dataset_path) / 'spacenet7' / 'sentinel2' / f'sentinel2_{aoi_id}.tif'
        visualization.plot_optical(ax_opt, opt_file)
        ax_opt.set_xlabel(f'(b) Optical (True Color)', fontsize=FONTSIZE)
        ax_opt.xaxis.set_label_coords(0.5, -0.025)

        ax_sn7 = axs[2]
        sn7_file = Path(dataset_path) / 'spacenet7' / 'buildings' / f'buildings_{aoi_id}.tif'
        visualization.plot_buildings(ax_sn7, sn7_file, 0)
        ax_sn7.set_xlabel(f'(c) Ground Truth', fontsize=FONTSIZE)
        ax_sn7.xaxis.set_label_coords(0.5, -0.025)

        ax_ours = axs[3]
        ours_file = Path(dataset_path) / 'spacenet7' / cfg.NAME / f'{cfg.NAME}_{aoi_id}.tif'
        visualization.plot_buildings(ax_ours, ours_file, 0.5)
        ax_ours.set_xlabel(f'(d) Ours Pred', fontsize=FONTSIZE)
        ax_ours.xaxis.set_label_coords(0.5, -0.025)

        ax_ours = axs[4]
        visualization.plot_buildings(ax_ours, ours_file, None)
        ax_ours.set_xlabel(f'(d) Ours Prob', fontsize=FONTSIZE)
        ax_ours.xaxis.set_label_coords(0.5, -0.025)

        plt.tight_layout()
        folder = Path(cfg.PATHS.OUTPUT) / 'plots' / 'qualitative_results' / cfg.NAME
        folder.mkdir(exist_ok=True)
        plot_file = folder / f'qualitative_comparison_{aoi_id}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)


def regional_ghs_comparison_histograms(cfg: experiment_manager.CfgNode):
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    region_names = get_region_names(cfg.PATHS.DATASET)
    for i, region in enumerate(region_names):
        ax_i = i // 3
        ax_j = i % 3
        ax = axs[ax_i, ax_j]

        # ghs
        data = get_quantitative_data(cfg.PATHS.OUTPUT, 'ghs')
        y_prob = np.concatenate([site['y_prob'] for site in data[region]]).flatten()
        weights = np.ones_like(y_prob) / len(y_prob)
        ax.hist(y_prob, weights=weights, bins=25, alpha=0.6, label='GHS-S2')

        # ours (cfg)
        data = get_quantitative_data(cfg.PATHS.OUTPUT, cfg.NAME)
        y_prob = np.concatenate([site['y_prob'] for site in data[region]]).flatten()
        weights = np.ones_like(y_prob) / len(y_prob)
        ax.hist(y_prob, weights=weights, bins=25, alpha=0.6, label='Fusion-DA')

        if ax_j == 0:
            ax.set_ylabel('Frequency (%)', fontsize=FONTSIZE)
        if ax_i == 1:
            ax.set_xlabel('CNN output probability', fontsize=FONTSIZE)

        domain = r'$\mathcal{S}$' if region == 'NWW' else r'$\mathcal{T}$'
        ax.text(0.1, 0.087, f'{region} ({domain})', fontsize=24)
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

    plt.tight_layout()
    plt.legend(frameon=False, handletextpad=0.5, columnspacing=0.8, handlelength=0.6)
    output_file = Path(cfg.PATHS.OUTPUT) / 'plots' / f'histogram_ghs_comparison_{cfg.NAME}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


# https://matplotlib.org/stable/gallery/statistics/boxplot_color.html
def regional_comparison_boxplots(metric: str, metric_name: str, cfg: experiment_manager.CfgNode, gap_index: int):
    config_names = ['sar', 'optical', 'fusion', cfg.NAME, 'ghs', 'wsf2019']
    names = ['SAR', 'Opt', 'Fus.', 'Fus.-DA', 'GHS', 'WSF']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2']

    box_width = 0.4
    def set_box_color(bp, color):
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    fig, axs = plt.subplots(2, 3, figsize=(16, 12))
    region_names = get_region_names(cfg.PATHS.DATASET)
    for i, region in enumerate(region_names):
        ax_i = i // 3
        ax_j = i % 3
        ax = axs[ax_i, ax_j]
        boxplot_data = []
        for j, config_name in enumerate(config_names):
            data = get_quantitative_data(cfg.PATHS.OUTPUT, config_name)
            region_data = [site[metric] for site in data[region]]
            boxplot_data.append(region_data)

        x_positions = np.arange(len(config_names))
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
                patch.set_facecolor(colors[i_patch])
            else:
                patch.set_facecolor('white')
        set_box_color(bplot, 'k')
        ax.set_xlim((-0.5, len(config_names) - 0.5))
        ax.set_ylim((0, 1))
        if ax_j == 0:
            ax.set_ylabel(metric_name, fontsize=FONTSIZE)
        domain = r'$\mathcal{S}$' if region == 'NWW' else r'$\mathcal{T}$'
        ax.text(-0.2, 0.87, f'{region} ({domain})', fontsize=24)
        ax.yaxis.grid(True)

        x_ticks = [(gap_index - 1) / 2] + [i for i in range(gap_index, len(config_names))]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(['Ours'] + [names[i] for i in range(gap_index, len(config_names))], fontsize=FONTSIZE)

    handles = [Patch(facecolor=colors[i], edgecolor=colors[i]) for i in range(gap_index)]
    axs[1, 0].legend(handles, names, loc='lower left', ncol=2, frameon=False, handletextpad=0.5,
                     columnspacing=0.8, handlelength=0.6, fontsize=FONTSIZE)
    plt.tight_layout()
    output_file = Path(cfg.PATHS.OUTPUT) / 'plots' / f'boxplots_{metric}_{cfg.NAME}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    args = parsers.testing_inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    # qualitative_testing(cfg)
    qualitative_sota_comparison(cfg)
    regional_ghs_comparison_histograms(cfg)
    # metrics = ['f1_score', 'precision', 'recall', 'iou']
    # metric_names = ['F1 score', 'Precision', 'Recall', 'IoU']
    regional_comparison_boxplots('f1_score', 'F1 score', cfg, 4)
