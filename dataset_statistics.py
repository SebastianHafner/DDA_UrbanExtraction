import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from utils import datasets, geofiles, parsers, experiment_manager
import numpy as np
from tqdm import tqdm
import torch
import json

COLOR_TRUE = '#ffffff'  # '#b11218'
COLOR_FALSE = '#000000'  # '#fff5f0'
COLOR_NA = 'lightgray'


def get_region_names(dataset_path: str) -> list:
    file = Path(dataset_path) / 'spacenet7' / 'spacenet7_regions.json'
    metadata_regions = geofiles.load_json(file)
    n_regions = len(metadata_regions['regions'].keys())
    region_names = [metadata_regions['regions'][str(i)] for i in range(n_regions)]
    return region_names


def pixelareakm2(n, resolution=10):
    # area in m2 of 1 pixel
    area_m2_pixel = resolution ** 2
    # area in m2 of all pixels
    area_m2 = area_m2_pixel * n
    # area in km2
    area_km2 = area_m2 / 1e6
    return area_km2


def run_train_validation_statistics(cfg: experiment_manager):
    training_dataset = datasets.UrbanExtractionDataset(cfg, 'training', no_augmentations=True)
    train_labeled = 0
    train_unlabeled = 0
    train_builtup = 0
    for index in tqdm(range(len(training_dataset))):
        item = training_dataset.__getitem__(index)
        label = item['y'].cpu()
        if item['is_labeled']:
            train_labeled += torch.numel(label)
            train_builtup += torch.sum(label).item()
        else:
            train_unlabeled += torch.numel(label)
    train_background = train_labeled - train_builtup
    print(f'labeled: {train_labeled} (builtup: {train_builtup}, bg: {train_background}) - unlabeled: {train_unlabeled}')

    validation_dataset = datasets.UrbanExtractionDataset(cfg, 'validation', no_augmentations=True,
                                                         include_unlabeled=False)
    val_labeled = 0
    val_builtup = 0
    for index in tqdm(range(len(validation_dataset))):
        item = training_dataset.__getitem__(index)
        label = item['y'].cpu()
        val_labeled += torch.numel(label)
        val_builtup += torch.sum(label).item()
    val_background = val_labeled - val_builtup
    print(f'labeled: {val_labeled} (builtup: {val_builtup}, bg: {val_background})')

    output_data = {
        'train_labeled': train_labeled,
        'train_builtup': train_builtup,
        'train_background': train_background,
        'train_unlabeled': train_unlabeled,
        'val_labeled': val_labeled,
        'val_builtup': val_builtup,
        'val_background': val_background
    }

    output_file = Path(cfg.PATHS.OUTPUT) / 'dataset' / f'train_validation_statistics_{cfg.NAME}.json'
    output_file.parent.mkdir(exist_ok=True)
    geofiles.write_json(output_file, output_data)


def plot_train_validation(cfg: experiment_manager):
    data_file = Path(cfg.PATHS.OUTPUT) / 'dataset' / f'train_validation_statistics_{cfg.NAME}.json'
    if not data_file.exists():
        run_train_validation_statistics(cfg)
    data = geofiles.load_json(data_file)

    mpl.rcParams.update({'font.size': 20})
    ypos = [0.5, 1, 1.5]
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 3.5))
    neg = ax.barh(ypos[-1], data['train_background'], width, label='Non-built-up', color=COLOR_FALSE, edgecolor='k')
    pos = ax.barh(ypos[-1], data['train_builtup'], width, label='Built-up', left=data['train_background'],
                  color=COLOR_TRUE, edgecolor='k')
    na = ax.barh(ypos[1], data['train_unlabeled'], width, label='N/A', color=COLOR_TRUE,
                 hatch='/', edgecolor='k')
    ax.barh(ypos[0], data['val_background'], width, label='Non-built-up', color=COLOR_FALSE, edgecolor='k')
    ax.barh(ypos[0], data['val_builtup'], width, label='Built-up', left=data['val_background'], color=COLOR_TRUE,
            edgecolor='k')

    ax.ticklabel_format(style='sci')
    ax.set_xlabel('Number of pixels')

    ax.set_yticks(ypos)
    ax.set_yticklabels(['Val. labeled', 'Train. unlabeled', 'Train. labeled'])
    ax.legend((neg, pos, na), ('Non-built-up', 'Built-up', 'N/A'), loc='lower right', ncol=3, frameon=False,
              handletextpad=0.8, columnspacing=1, handlelength=1)
    out_file = Path(cfg.PATHS.OUTPUT) / 'plots' / 'dataset_stats' / 'train_val_dataset.jpeg'
    out_file.parent.mkdir(exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches='tight', format='jpeg')
    plt.close(fig)


def show_validation_training(cfg: experiment_manager.CfgNode):
    data_file = Path(cfg.PATHS.OUTPUT) / 'dataset' / f'train_validation_statistics_{cfg.NAME}.json'
    if not data_file.exists():
        run_train_validation_statistics(cfg)
    data = geofiles.load_json(data_file)

    # train labeled
    train_true = data['train_builtup']
    train_false = data['train_background']
    train_total = train_true + train_false
    area_total = pixelareakm2(train_total)
    percentage = train_true / (train_true + train_false) * 100
    print(f'Train (labeled): {percentage:.0f} % ({train_total:.2E}, {area_total:.0f})')

    # train unlabeled
    train_unlabeled = data['train_unlabeled']
    area_total = pixelareakm2(train_unlabeled)
    print(f'Train (unlabeled): {train_unlabeled:.2E}, {area_total:.0f}')

    # validation
    val_true = data['val_builtup']
    val_false = data['val_background']
    val_total = val_true + val_false
    area_total = pixelareakm2(val_total)
    percentage = val_true / (val_true + val_false) * 100
    print(f'Validation: {percentage:.0f} % ({val_total:.2E}, {area_total:.0f})')


def run_test_statistics(cfg: experiment_manager.CfgNode):
    test_dataset = datasets.SpaceNet7Dataset(cfg)
    data = {}
    for index in tqdm(range(len(test_dataset))):
        item = test_dataset.__getitem__(index)
        label = item['y'].cpu()
        region_name = item['region']
        labeled = torch.numel(label)
        builtup = torch.sum(label).item()
        if not region_name in data.keys():
            data[region_name] = {
                'labeled': 0,
                'builtup': 0,
                'background': 0
            }
        data[region_name]['labeled'] += labeled
        data[region_name]['builtup'] += builtup
        data[region_name]['background'] += (labeled - builtup)

    output_file = Path(cfg.PATHS.OUTPUT) / 'dataset' / f'test_statistics_{cfg.NAME}.json'
    geofiles.write_json(output_file, data)


def plot_test(cfg: experiment_manager.CfgNode):
    data_file = Path(cfg.PATHS.OUTPUT) / 'dataset' / f'test_statistics_{cfg.NAME}.json'
    if not data_file.exists():
        run_test_statistics(cfg)
    data = geofiles.load_json(data_file)

    mpl.rcParams.update({'font.size': 20})
    width = 0.5
    ypos = np.arange(len(data.keys()))
    fig, ax = plt.subplots(figsize=(10, 5))

    regions = get_region_names(cfg.PATHS.DATASET)
    n = len(regions) - 1
    for i, region_name in enumerate(regions):
        region_data = data[region_name]
        neg = ax.barh(n - i, region_data['background'], width, label='Non-built-up', color=COLOR_FALSE, edgecolor='k')
        pos = ax.barh(n - i, region_data['builtup'], width, label='Built-up', left=region_data['background'],
                      color=COLOR_TRUE, edgecolor='k')

    ax.ticklabel_format(style='sci')
    ax.set_xlabel(r'Number of pixels')

    ax.set_yticks(ypos)
    y_ticklabels = ['Source' if region == 'NWW' else f'Target {region}' for region in regions][::-1]
    ax.set_yticklabels(y_ticklabels)
    ax.legend((neg, pos), ('Non-built-up', 'Built-up'), ncol=1, frameon=False,
              handletextpad=0.8, columnspacing=1, handlelength=1)
    out_file = Path(cfg.PATHS.OUTPUT) / 'plots' / 'dataset_stats' / 'test_dataset.jpeg'
    out_file.parent.mkdir(exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches='tight', format='jpeg')
    plt.close(fig)


def show_test(cfg: experiment_manager.CfgNode):
    data_file = Path(cfg.PATHS.OUTPUT) / 'dataset' / f'test_statistics_{cfg.NAME}.json'
    if not data_file.exists():
        run_test_statistics(cfg)
    data = geofiles.load_json(data_file)

    total = 0
    for region_name in data.keys():
        region_data = data[region_name]
        true = region_data['builtup']
        false = region_data['background']
        region_total = true + false
        total += region_total
        percentage = true / (true + false) * 100
        print(f'{region_name}: {percentage:.0f} % BUA ({region_total:.2E} pixels)')
    total_area = pixelareakm2(total)
    print(f'Test {total:.2E} {total_area:.0f}')


if __name__ == '__main__':
    args = parsers.testing_inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    # training validation
    plot_train_validation(cfg)
    # show_validation_training(cfg)
    # testing
    plot_test(cfg)
    # show_test(cfg)