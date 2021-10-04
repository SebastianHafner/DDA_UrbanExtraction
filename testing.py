from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils import metrics, geofiles, experiment_manager, networks, datasets, paths, visualization
from sklearn.metrics import precision_recall_curve

FONTSIZE = 20
mpl.rcParams.update({'font.size': FONTSIZE})


def run_inference(config_name: str, site: str):
    print(f'running inference for {site} with {config_name}...')

    dirs = paths.load_paths()

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
    output_file = Path(dirs.OUTPUT) / 'inference' / config_name / f'prob_{site}_{config_name}.tif'
    output_file.parent.mdkir(exist_ok=True)
    geofiles.write_tif(output_file, prob_output, transform, crs)


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


def plot_precision_recall_curve(site: str, config_names: list, names: list = None, show_legend: bool = False,
                                save_plot: bool = False):

    dirs = paths.load_paths()

    fig, ax = plt.subplots()

    # getting data and if not available produce
    for i, config_name in enumerate(config_names):
        data_file_config = Path(dirs.OUTPUT) / 'quantitative_evaluation' / config_name / f'{site}_{config_name}.npy'
        if not data_file_config.exists():
            run_quantitative_evaluation(config_name, site, save_output=True)
        data_config = np.load(data_file_config)
        prec, rec, thresholds = precision_recall_curve(data_config[0,], data_config[1,])
        label = config_name if names is None else names[i]
        ax.plot(rec, prec, label=label)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.set_xlabel('Recall', fontsize=FONTSIZE)
        ax.set_ylabel('Precision', fontsize=FONTSIZE)
        ax.set_aspect('equal', adjustable='box')
        ticks = np.linspace(0, 1, 6)
        tick_labels = [f'{tick:.1f}' for tick in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=FONTSIZE)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels, fontsize=FONTSIZE)
        if show_legend:
            ax.legend()
    if save_plot:
        plot_file = Path(dirs.OUTPUT) / 'plots' / 'precision_recall_curve' / f'{site}_{"".join(config_names)}.png'
        plot_file.parent.mkdir(exist_ok=True)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# SpaceNet7 testing

GROUPS = [(1, 'NA_AU', '#63cd93'), (2, 'SA', '#f0828f'), (3, 'EU', '#6faec9'), (4, 'SSA', '#5f4ad9'),
          (5, 'NAF_ME', '#8dee47'), (6, 'AS', '#d9b657'), ('total', 'Total', '#ffffff')]
GROUP_NAMES = ['NA_AU', 'SA', 'EU', 'SSA', 'NAF_ME', 'AS']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def run_quantitative_inference_sn7(config_name: str):
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
            if aoi_id == 'L15-0434E-1218N_1736_3318_13':
                continue
            img = test_site['x'].to(device)
            y_prob = net(img.unsqueeze(0))
            y_prob = torch.sigmoid(y_prob).flatten().cpu().numpy()
            y_true = test_site['y'].flatten().cpu().numpy()

            group_name = test_site['group_name']
            if group_name not in data.keys():
                data[group_name] = []

            f1 = metrics.f1_score_from_prob(y_prob, y_true, 0.5)
            p = metrics.precsision_from_prob(y_prob, y_true, 0.5)
            r = metrics.recall_from_prob(y_prob, y_true, 0.5)

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
            run_quantitative_inference_sn7(config_name)
        else:
            raise Exception('No data and not allowed to run quantitative inference!')
    # run_quantitative_inference(config_name)
    data = np.load(data_file, allow_pickle=True)
    data = dict(data[()])
    return data


def show_quantitative_testing_sn7(config_name: str):
    print(f'{"-" * 10} {config_name} {"-" * 10}')

    data = get_quantitative_data_sn7(config_name)
    for metric in ['f1_score', 'precision', 'recall']:
        print(metric)
        region_values = []
        for region_name, region in data.items():

            y_true = np.concatenate([site['y_true'] for site in region], axis=0)
            y_prob = np.concatenate([site['y_prob'] for site in region], axis=0)

            if metric == 'f1_score':
                value = metrics.f1_score_from_prob(y_prob, y_true, 0.5)
            elif metric == 'precision':
                value = metrics.precsision_from_prob(y_prob, y_true, 0.5)
            else:
                value = metrics.recall_from_prob(y_prob, y_true, 0.5)

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


def plot_boxplots_sn7(config_names: list, names: list = None):
    mpl.rcParams.update({'font.size': 16})
    metrics = ['f1_score', 'precision', 'recall']
    metric_names = ['F1 score', 'Precision', 'Recall']
    metric_chars = ['A', 'B', 'C']
    box_width = 0.2
    wisk_width = 0.1
    line_width = 2
    point_size = 40

    for m, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        fig, ax = plt.subplots(figsize=(12, 5))

        def custom_boxplot(x_pos: float, values: list, color: str, annotation: str = None):
            min_ = np.min(values)
            max_ = np.max(values)
            std = np.std(values)
            mean = np.mean(values)
            median = np.median(values)
            min_ = mean - std
            max_ = mean + std

            line_kwargs = {'c': color, 'lw': line_width}
            point_kwargs = {'c': color, 's': point_size}

            # vertical line
            ax.plot(2 * [x_pos], [min_, max_], **line_kwargs)

            # whiskers
            x_positions = [x_pos - wisk_width / 2, x_pos + wisk_width / 2]
            ax.plot(x_positions, [min_, min_], **line_kwargs)
            ax.plot(x_positions, [max_, max_], **line_kwargs)

            # median
            ax.scatter([x_pos], [mean], **point_kwargs)

            if annotation is not None:
                ax.text(-0.07, 0.12, annotation, fontsize=40)

            pass

        for i, config_name in enumerate(config_names):

            data = get_quantitative_data_sn7(config_name)

            for j, group_name in enumerate(GROUP_NAMES):
                values = [site[metric] for site in data[group_name]]
                x_pos = j + (i * box_width) + j * 0.1
                custom_boxplot(x_pos, values, color=COLORS[i], annotation=metric_chars[m])

        ax.set_ylim((0, 1))
        ax.set_ylabel(metric_name)

        x_ticks = np.arange(len(GROUP_NAMES)) + (len(config_names) - 1) * box_width / 2
        x_ticks = [x_tick + i * 0.1 for i, x_tick in enumerate(x_ticks)]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(GROUP_NAMES)

        if metric == 'f1_score':
            handles = [Line2D([0], [0], color=COLORS[i], lw=line_width) for i in range(len(config_names))]
            ax.legend(handles, names, loc='upper center', ncol=4, frameon=False, handletextpad=0.8,
                      columnspacing=1, handlelength=1)
        plt.show()
        plt.close(fig)


def plot_activation_comparison_sn7(config_names: list, save_plots: bool = False):

    dirs = paths.load_paths()

    # setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = [experiment_manager.load_cfg(config_name) for config_name in config_names]
    dataset_collection = [datasets.SpaceNet7Dataset(cfg) for cfg in configs]

    # optical, sar, reference and predictions (n configs)
    n_plots = 3 + len(config_names)

    for index in range(len(dataset_collection[0])):
        fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * 3, 4))
        for i, (cfg, dataset) in enumerate(zip(configs, dataset_collection)):

            net, _, _ = networks.load_checkpoint(cfg.INFERENCE.CHECKPOINT, cfg, device)
            net.eval()

            sample = dataset.__getitem__(index)
            aoi_id = sample['aoi_id']
            country = sample['country']
            group_name = sample['group_name']

            if i == 0:
                fig.subplots_adjust(wspace=0, hspace=0)
                mpl.rcParams['axes.linewidth'] = 1

                optical_file = Path(dirs.DATASET) / 'sn7' / 'sentinel2' / f'sentinel2_{aoi_id}.tif'
                visualization.plot_optical(axs[0], optical_file, vis='true_color', show_title=False)

                sar_file = Path(dirs.DATASET) / 'sn7' / 'sentinel1' / f'sentinel1_{aoi_id}.tif'
                visualization.plot_sar(axs[1], sar_file, show_title=False)

                label = cfg.DATALOADER.LABEL
                label_file = Path(dirs.DATASET) / 'sn7' / label / f'{label}_{aoi_id}.tif'
                visualization.plot_buildings(axs[2], label_file, show_title=False)

            with torch.no_grad():
                x = sample['x'].to(device)
                logits = net(x.unsqueeze(0))
                prob = torch.sigmoid(logits[0, 0,])
                prob = prob.detach().cpu().numpy()
                visualization.plot_probability(axs[3 + i], prob)

        title = f'{country} ({group_name})'

        axs[0].set_ylabel(title, fontsize=16)
        if save_plots:
            folder =  Path(dirs.OUTPUT) / 'plots' / 'testing' / f'qualitative ' + '_'.join(config_names)
            folder.mkdir(exist_ok=True)
            file = folder / f'{aoi_id}.png'
            plt.savefig(file, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


def plot_activation_comparison_assembled_sn7(config_names: list, names: list, aoi_ids: list = None,
                                         save_plot: bool = False):

    dirs = paths.load_paths()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mpl.rcParams['axes.linewidth'] = 1

    # setting up plot
    plot_size = 3
    plot_rows = len(aoi_ids)
    plot_height = plot_size * plot_rows
    plot_cols = 3 + len(config_names)  # optical, sar, reference and predictions (n configs)
    plot_width = plot_size * plot_cols
    fig, axs = plt.subplots(plot_rows, plot_cols, figsize=(plot_width, plot_height))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    for i, config_name in enumerate(config_names):

        # loading configs, datasets, and networks
        cfg = experiment_manager.load_cfg(config_name)
        dataset = datasets.SpaceNet7Dataset(cfg)
        net, _, _ = networks.load_checkpoint(cfg.INFERENCE.CHECKPOINT, cfg, device)
        net.eval()

        for j, aoi_id in enumerate(aoi_ids):
            index = dataset.get_index(aoi_id)
            sample = dataset.__getitem__(index)
            country = sample['country']
            if country == 'United States':
                country = 'US'
            if country == 'United Kingdom':
                country = 'UK'
            if country == 'Saudi Arabia':
                country = 'Saudi Ar.'
            group_name = sample['group_name']

            optical_file = Path(dirs.DATASET) / 'sn7' / 'sentinel2' / f'sentinel2_{aoi_id}.tif'
            visualization.plot_optical(axs[j, 0], optical_file, vis='true_color', show_title=False)
            axs[-1, 0].set_xlabel('(a) Image Optical', fontsize=FONTSIZE)

            sar_file = Path(dirs.DATASET) / 'sn7' / 'sentinel1' / f'sentinel1_{aoi_id}.tif'
            visualization.plot_sar(axs[j, 1], sar_file, show_title=False)
            axs[-1, 1].set_xlabel('(b) Image SAR', fontsize=FONTSIZE)

            label = cfg.DATALOADER.LABEL
            label_file = Path(dirs.DATASET) / 'sn7' / label / f'{label}_{aoi_id}.tif'
            visualization.plot_buildings(axs[j, 2], label_file, show_title=False)
            axs[-1, 2].set_xlabel('(c) Ground Truth', fontsize=FONTSIZE)

            with torch.no_grad():
                x = sample['x'].to(device)
                logits = net(x.unsqueeze(0))
                prob = torch.sigmoid(logits.squeeze())
                prob = prob.cpu().numpy()
                ax = axs[j, 3 + i]
                visualization.plot_probability(ax, prob)

            if i == 0:  # row labels only need to be set once
                row_label = f'{country} ({group_name})'
                axs[j, 0].set_ylabel(row_label, fontsize=FONTSIZE)

            col_letter = chr(ord('a') + 3 + i)
            col_label = f'({col_letter}) {names[i]}'
            axs[-1, 3 + i].set_xlabel(col_label, fontsize=FONTSIZE)

    if save_plot:
        folder = Path(dirs.OUTPUT) / 'plots' / 'testing' / 'qualitative'
        folder.mkdir(exist_ok=True)
        file = folder / f'qualitative_results_assembled.png'
        plt.savefig(file, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_precision_recall_curve_sn7(config_names: list, names: list = None, show_legend: bool = False,
                                    save_plot: bool = False):

    dirs = paths.load_paths()

    # getting data and if not available produce
    for i, region_name in enumerate(GROUP_NAMES):
        fig, ax = plt.subplots()
        for i, config_name in enumerate(config_names):
            data = get_quantitative_data_sn7(config_name)
            y_true = np.concatenate([site['y_true'] for site in data[region_name]], axis=0).astype(np.int)
            y_prob = np.concatenate([site['y_prob'] for site in data[region_name]], axis=0)
            prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
            label = config_name if names is None else names[i]
            ax.plot(rec, prec, label=label)

        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.set_xlabel('Recall', fontsize=FONTSIZE)
        ax.set_ylabel('Precision', fontsize=FONTSIZE)
        ax.set_aspect('equal', adjustable='box')
        ticks = np.linspace(0, 1, 6)
        tick_labels = [f'{tick:.1f}' for tick in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=FONTSIZE)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels, fontsize=FONTSIZE)
        if show_legend:
            ax.legend()

        if save_plot:
            plot_file = Path(dirs.OUTPUT) / 'plots' / 'precision_recall_curve' / f'{region_name}_{"".join(config_names)}.png'
            plot_file.parent.mkdir(exist_ok=True)
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close(fig)


if __name__ == '__main__':

    config_name = 'sar'
    city = 'calgary'

    config_names = ['sar', 'optical']
    names = ['SAR', 'Optical']

    aoi_ids = [
        'L15-0357E-1223N_1429_3296_13',  # na
        'L15-0614E-0946N_2459_4406_13',  # sa
        'L15-1014E-1375N_4056_2688_13',  # eu
        'L15-0924E-1108N_3699_3757_13',  # ssa
        'L15-1015E-1062N_4061_3941_13',  # ssa
        'L15-0977E-1187N_3911_3441_13',  # naf_me
        'L15-1204E-1202N_4816_3380_13',  # naf_me
        'L15-1439E-1134N_5759_3655_13',  # as
        'L15-1716E-1211N_6864_3345_13',  # as
    ]

    # produce_label(config_name, city)
    # run_inference(config_name, city)


    # SpaceNet7 stuff

    # run_quantitative_inference_sn7(config_name)
    # show_quantitative_testing_sn7(config_name)
    # plot_boxplots_sn7(config_names, names)

    # plot_activation_comparison_sn7(config_names, save_plots=False)
    # plot_activation_comparison_assembled_sn7(config_names, names, aoi_ids, save_plot=False)

    plot_precision_recall_curve_sn7(config_names, names, save_plot=False)
    # plot_threshold_dependency(config_names, names)


