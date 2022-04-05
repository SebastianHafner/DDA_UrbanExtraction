from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, auc
from utils import geofiles, experiment_manager, visualization, parsers, metrics
from scipy.interpolate import interp1d

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


# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
def regional_comparison_roc_analysis(cfg: experiment_manager.CfgNode):
    config_names = [cfg.NAME, 'ghs']
    names = ['Fus.-DA', 'GHS']
    colors = ['#d62728', '#9467bd', '#e377c2']

    fig, axs = plt.subplots(2, 3, figsize=(16, 12))
    region_names = get_region_names(cfg.PATHS.DATASET)
    for i, region in enumerate(region_names):
        ax_i = i // 3
        ax_j = i % 3
        ax = axs[ax_i, ax_j]
        for j, config_name in enumerate(config_names):
            data = get_quantitative_data(cfg.PATHS.OUTPUT, config_name)
            y_prob = np.concatenate([aoi['y_prob'] for aoi in data[region]])
            y_true = np.concatenate([aoi['y_true'] for aoi in data[region]])

            auc = roc_auc_score(y_true, y_prob)

            # summarize scores
            print('ROC AUC=%.3f' % (auc))
            # calculate roc curves
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            # plot the roc curve for the model
            ax.plot(fpr, tpr, linestyle='-', label=names[j], c=colors[j])

        domain = r'$\mathcal{S}$' if region == 'NWW' else r'$\mathcal{T}$'
        ax.text(0.6, 0.1, f'{region} ({domain})', fontsize=24)

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ticks = np.linspace(0, 1, 6, endpoint=True)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        tick_labels = [f'{tick:.1f}' for tick in ticks]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=FONTSIZE)
        ax.set_yticklabels(tick_labels, fontsize=FONTSIZE)
        ax.set_aspect('equal')

    handles = [Patch(facecolor=colors[i], edgecolor=colors[i]) for i in range(len(config_names))]
    axs[0, 0].legend(handles, names, loc='center right', ncol=1, frameon=False, handletextpad=0.5,
                     columnspacing=0.8, handlelength=0.6, fontsize=FONTSIZE)
    plt.tight_layout()
    output_file = Path(cfg.PATHS.OUTPUT) / 'plots' / f'roccurves_{cfg.NAME}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)



# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
def regional_comparison_roc_analysis_v2(cfg: experiment_manager.CfgNode, n_thresholds: int = 1_000):
    config_names = [cfg.NAME, 'ghs']
    names = ['Fus.-DA', 'GHS']
    colors = ['#d62728', '#9467bd', '#e377c2']
    thresholds = np.linspace(0.01, 1, 1_000)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for j, config_name in enumerate(config_names):
        i = 0
        fpr_all = np.empty((60, len(thresholds)), dtype=np.float32)
        tpr_all = np.empty((60, len(thresholds)), dtype=np.float32)
        region_names = get_region_names(cfg.PATHS.DATASET)
        for region in region_names:
            data = get_quantitative_data(cfg.PATHS.OUTPUT, config_name)
            for aoi_data in data[region]:
                # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                y_prob, y_true = aoi_data['y_prob'], aoi_data['y_true']
                #    calculate roc curves
                # fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, 1_000)
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob, drop_intermediate=False)
                axs[j].plot(fpr, tpr, linestyle='-', label=names[j], c=colors[j], lw=0.5)


                f = interp1d(roc_thresholds, fpr, kind='nearest')
                fpr = f(thresholds)
                fpr_all[i, :] = fpr
                f = interp1d(roc_thresholds, tpr, kind='nearest')
                tpr = f(thresholds)
                tpr_all[i, :] = tpr


                # plot the roc curve for the model
                i += 1

        mean_fpr = np.mean(fpr_all, axis=0)
        mean_tpr = np.mean(tpr_all, axis=0)
        # axs[j].plot(mean_fpr, mean_tpr, linestyle='-', label=names[j], c=colors[j])
        # axs[j].fill_between(t, lower_bound, upper_bound, facecolor='yellow', alpha=0.5, label='1 sigma range')

    for _, ax in np.ndenumerate(axs):
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ticks = np.linspace(0, 1, 6, endpoint=True)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        tick_labels = [f'{tick:.1f}' for tick in ticks]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=FONTSIZE)
        ax.set_yticklabels(tick_labels, fontsize=FONTSIZE)
        ax.set_aspect('equal')

    handles = [Patch(facecolor=colors[i], edgecolor=colors[i]) for i in range(len(config_names))]
    axs[0].legend(handles, names, loc='center right', ncol=1, frameon=False, handletextpad=0.5,
                     columnspacing=0.8, handlelength=0.6, fontsize=FONTSIZE)
    plt.tight_layout()
    output_file = Path(cfg.PATHS.OUTPUT) / 'plots' / f'roccurves_v2_{cfg.NAME}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
def boxplot_auc_comparison(cfg: experiment_manager.CfgNode):
    config_names = [cfg.NAME, 'ghs']
    names = ['Fus.-DA', 'GHS']
    colors = ['#d62728', '#9467bd', '#e377c2']
    boxplot_data = [[] for _ in range(len(config_names))]

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    region_names = get_region_names(cfg.PATHS.DATASET)
    for i, region in enumerate(region_names):
        for j, config_name in enumerate(config_names):
            data = get_quantitative_data(cfg.PATHS.OUTPUT, config_name)
            for aoi_data in data[region]:
                y_prob = aoi_data['y_prob']
                y_true = aoi_data['y_true']

                auc = roc_auc_score(y_true, y_prob)
                boxplot_data[j].append(auc)

    bplot = ax.boxplot(boxplot_data, whis=[0, 100])

    def set_box_color(bp, color):
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    for i_patch, patch in enumerate(bplot['boxes']):
        patch.set_facecolor(colors[i_patch])

    ax.set_ylabel('AUC')
    ticks = np.linspace(0, 1, 6, endpoint=True)
    ax.set_ylim((0, 1))
    tick_labels = [f'{tick:.1f}' for tick in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=FONTSIZE)
    ax.set_xticklabels(names, fontsize=FONTSIZE)
    plt.tight_layout()
    output_file = Path(cfg.PATHS.OUTPUT) / 'plots' / f'boxplot_auc_{cfg.NAME}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
def regional_comparison_pr_curves(cfg: experiment_manager.CfgNode):
    config_names = [cfg.NAME, 'ghs']
    names = ['Fus.-DA', 'GHS']
    colors = ['#d62728', '#9467bd', '#e377c2']

    fig, axs = plt.subplots(2, 3, figsize=(16, 12))
    region_names = get_region_names(cfg.PATHS.DATASET)
    for i, region in enumerate(region_names):
        ax_i = i // 3
        ax_j = i % 3
        ax = axs[ax_i, ax_j]
        for j, config_name in enumerate(config_names):
            data = get_quantitative_data(cfg.PATHS.OUTPUT, config_name)
            for aoi_data in data[region]:
                aoi_id = aoi_data['aoi_id']
                y_prob = aoi_data['y_prob']
                y_true = aoi_data['y_true']

                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                # auc = auc(recall, precision)

                ax.plot(recall, precision, linestyle='-', label=names[j], c=colors[j])

        domain = r'$\mathcal{S}$' if region == 'NWW' else r'$\mathcal{T}$'
        ax.text(0.6, 0.1, f'{region} ({domain})', fontsize=24)

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ticks = np.linspace(0, 1, 6, endpoint=True)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        tick_labels = [f'{tick:.1f}' for tick in ticks]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=FONTSIZE)
        ax.set_yticklabels(tick_labels, fontsize=FONTSIZE)
        ax.set_aspect('equal')

    handles = [Patch(facecolor=colors[i], edgecolor=colors[i]) for i in range(len(config_names))]
    axs[0, 0].legend(handles, names, loc='center right', ncol=1, frameon=False, handletextpad=0.5,
                     columnspacing=0.8, handlelength=0.6, fontsize=FONTSIZE)
    plt.tight_layout()
    output_file = Path(cfg.PATHS.OUTPUT) / 'plots' / f'prcurves_{cfg.NAME}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
def regional_comparison_pr_curves_v2(cfg: experiment_manager.CfgNode):
    config_names = [cfg.NAME, 'ghs']
    names = ['Fus.-DA', 'GHS']
    colors = ['#d62728', '#9467bd', '#e377c2']
    thresholds = np.linspace(0, 1, 1_000, endpoint=True)
    fig, axs = plt.subplots(2, 3, figsize=(16, 12))
    region_names = get_region_names(cfg.PATHS.DATASET)
    for i, region in enumerate(region_names):
        ax_i = i // 3
        ax_j = i % 3
        ax = axs[ax_i, ax_j]
        for j, config_name in enumerate(config_names):
            data = get_quantitative_data(cfg.PATHS.OUTPUT, config_name)
            precision_list, recall_list = [], []
            for aoi_data in data[region]:
                y_prob = aoi_data['y_prob']
                y_true = aoi_data['y_true']

                precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
                pr_thresholds = np.concatenate((np.array([0]), pr_thresholds, np.array([1])), axis=0)
                recall = np.concatenate((np.array([1]), recall), axis=0)
                precision = np.concatenate((np.array([0]), precision), axis=0)
                f = interp1d(pr_thresholds, precision, kind='nearest')
                precision_list.append(f(thresholds))
                f = interp1d(pr_thresholds, recall, kind='nearest')
                recall_list.append(f(thresholds))

            mean_precision = np.mean(np.stack(precision_list), axis=0)
            mean_recall = np.mean(np.stack(recall_list), axis=0)
            ax.plot(mean_recall, mean_precision, linestyle='-', label=names[j], c=colors[j])

        title = 'Source' if region == 'NWW' else f'Target {region}'
        ax.text(0.1, 0.1, title, fontsize=24)

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ticks = np.linspace(0, 1, 6, endpoint=True)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        tick_labels = [f'{tick:.1f}' for tick in ticks]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=FONTSIZE)
        ax.set_yticklabels(tick_labels, fontsize=FONTSIZE)
        ax.set_aspect('equal')

    handles = [Patch(facecolor=colors[i], edgecolor=colors[i]) for i in range(len(config_names))]
    axs[0, 0].legend(handles, names, loc='center left', ncol=1, frameon=False, handletextpad=0.5,
                     columnspacing=0.8, handlelength=0.6, fontsize=FONTSIZE)
    plt.tight_layout()
    output_file = Path(cfg.PATHS.OUTPUT) / 'plots' / f'prcurves_v2_{cfg.NAME}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
def boxplot_pr_auc_comparison(cfg: experiment_manager.CfgNode):
    config_names = [cfg.NAME, 'ghs']
    names = ['Fusion-DA', 'GHS']
    colors = ['#d62728', '#9467bd', '#e377c2']
    boxplot_data = [[] for _ in range(len(config_names))]
    f1_scores = [[] for _ in range(len(config_names))]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    region_names = get_region_names(cfg.PATHS.DATASET)
    for i, region in enumerate(region_names):
        for j, config_name in enumerate(config_names):
            data = get_quantitative_data(cfg.PATHS.OUTPUT, config_name)
            for aoi_data in data[region]:
                y_prob = aoi_data['y_prob']
                y_true = aoi_data['y_true']
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                pr_auc = auc(recall, precision)
                boxplot_data[j].append(pr_auc)
                y_pred = aoi_data['y_pred']
                f1_scores[j].append(metrics.f1_score_from_prob(y_pred, y_true))


    bplot = ax.boxplot(boxplot_data, whis=[5, 95], patch_artist=True)
    ours = np.array(boxplot_data[0])
    theirs = np.array(boxplot_data[1])
    print(np.sum(theirs > ours))

    ours = np.array(f1_scores[0])
    theirs = np.array(f1_scores[1])
    print(np.sum(theirs > ours))

    def set_box_color(bp, color):
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    for i_patch, patch in enumerate(bplot['boxes']):
        patch.set_facecolor(colors[i_patch])

    set_box_color(bplot, 'k')

    ax.set_ylabel('Area under precision-recall curve', fontsize=FONTSIZE)
    ticks = np.linspace(0, 1, 6, endpoint=True)
    ax.set_ylim((0, 1))
    tick_labels = [f'{tick:.1f}' for tick in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=FONTSIZE)
    ax.set_xticklabels(names, fontsize=FONTSIZE)
    plt.tight_layout()
    output_file = Path(cfg.PATHS.OUTPUT) / 'plots' / f'boxplot_pr_auc_{cfg.NAME}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)





if __name__ == '__main__':
    args = parsers.testing_inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    # qualitative_testing(cfg)
    # qualitative_sota_comparison(cfg)
    # regional_ghs_comparison_histograms(cfg)
    # regional_comparison_roc_analysis_v2(cfg)
    regional_comparison_pr_curves_v2(cfg)
    # boxplot_pr_auc_comparison(cfg)
    # boxplot_auc_comparison(cfg)
    # metrics = ['f1_score', 'precision', 'recall', 'iou']
    # metric_names = ['F1 score', 'Precision', 'Recall', 'IoU']
    # regional_comparison_boxplots('f1_score', 'F1 score', cfg, 4)
