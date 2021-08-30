import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from utils.datasets import UrbanExtractionDataset, SpaceNet7Dataset
from experiment_manager.config import config
from utils.geotiff import *
import numpy as np
from tqdm import tqdm
import torch
import json

ROOT_PATH = Path('/storage/shafner/urban_extraction')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')

COLOR_TRUE = '#ffffff'  # '#b11218'
COLOR_FALSE = '#000000'  # '#fff5f0'
COLOR_NA = 'lightgray'


def pixelareakm2(n, resolution=10):
    # area in m2 of 1 pixel
    area_m2_pixel = resolution**2
    # area in m2 of all pixels
    area_m2 = area_m2_pixel * n
    # area in km2
    area_km2 = area_m2 / 1e6
    return area_km2


def run_train_validation_statistics(config_name: str):
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')

    training_dataset = UrbanExtractionDataset(cfg, 'training', no_augmentations=True)
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

    validation_dataset = UrbanExtractionDataset(cfg, 'validation', no_augmentations=True, include_unlabeled=False)
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

    output_file = ROOT_PATH / 'plots' / 'dataset' / f'train_validation_statistics_{config_name}.json'
    with open(str(output_file), 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)


def plot_train_validation(config_name: str):

    data_file = ROOT_PATH / 'plots' / 'dataset' / f'train_validation_statistics_{config_name}.json'
    if not data_file.exists():
        run_train_validation_statistics(config_name)
    data = load_json(data_file)

    mpl.rcParams.update({'font.size': 20})
    ypos = [0.5, 1, 1.5]
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 3.5))
    neg = ax.barh(ypos[-1], data['train_background'], width, label='False', color=COLOR_FALSE, edgecolor='k')
    pos = ax.barh(ypos[-1], data['train_builtup'], width, label='True', left=data['train_background'],
                  color=COLOR_TRUE, edgecolor='k')
    na = ax.barh(ypos[1], data['train_unlabeled'], width, label='N/A', color=COLOR_TRUE,
                 hatch='/', edgecolor='k')
    ax.barh(ypos[0], data['val_background'], width, label='False', color=COLOR_FALSE, edgecolor='k')
    ax.barh(ypos[0], data['val_builtup'], width, label='True', left=data['val_background'], color=COLOR_TRUE,
            edgecolor='k')

    ax.ticklabel_format(style='sci')
    ax.set_xlabel('Number of Pixels')

    ax.set_yticks(ypos)
    ax.set_yticklabels(['Validation', 'Train unlabeled', 'Train labeled'])
    ax.legend((neg, pos, na), ('False', 'True', 'N/A'), loc='lower right', ncol=3, frameon=False,
              handletextpad=0.8, columnspacing=1, handlelength=1)
    plt.show()


def show_validation_training(config_name: str):

    data_file = ROOT_PATH / 'plots' / 'dataset' / f'train_validation_statistics_{config_name}.json'
    if not data_file.exists():
        run_train_validation_statistics(config_name)
    data = load_json(data_file)

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


def run_test_statistics(config_name: str):
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')

    test_dataset = SpaceNet7Dataset(cfg)
    data = {}
    for index in tqdm(range(len(test_dataset))):
        item = test_dataset.__getitem__(index)
        label = item['y'].cpu()
        group_name = item['group_name']
        labeled = torch.numel(label)
        builtup = torch.sum(label).item()
        if not group_name in data.keys():
            data[group_name] = {
                'labeled': 0,
                'builtup': 0,
                'background': 0
            }
        data[group_name]['labeled'] += labeled
        data[group_name]['builtup'] += builtup
        data[group_name]['background'] += (labeled - builtup)

    output_file = ROOT_PATH / 'plots' / 'dataset' / f'test_statistics_{config_name}.json'
    with open(str(output_file), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def plot_test(config_name: str):

    data_file = ROOT_PATH / 'plots' / 'dataset' / f'test_statistics_{config_name}.json'
    if not data_file.exists():
        run_test_statistics(config_name)
    data = load_json(data_file)

    mpl.rcParams.update({'font.size': 20})
    width = 0.5
    ypos = np.arange(len(data.keys()))
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (site, site_data) in enumerate(data.items()):
        site_data = data[site]
        neg = ax.barh(i, site_data['background'], width, label='False', color=COLOR_FALSE, edgecolor='k')
        pos = ax.barh(i, site_data['builtup'], width, label='True', left=site_data['background'],
                      color=COLOR_TRUE, edgecolor='k')

    ax.ticklabel_format(style='sci')
    ax.set_xlabel('Number of Pixels')

    ax.set_yticks(ypos)
    ax.set_yticklabels(data.keys())
    ax.legend((neg, pos), ('False', 'True'), ncol=1, frameon=False,
              handletextpad=0.8, columnspacing=1, handlelength=1)
    plt.show()


def show_test(config_name: str):

    data_file = ROOT_PATH / 'plots' / 'dataset' / f'test_statistics_{config_name}.json'
    if not data_file.exists():
        run_test_statistics(config_name)
    data = load_json(data_file)

    total = 0
    for group_name in data.keys():
        group_data = data[group_name]
        true = group_data['builtup']
        false = group_data['background']
        group_total = true + false
        total += group_total
        percentage = true / (true + false) * 100
        print(f'{group_name}: {percentage:.0f} % BUA ({group_total:.2E} pixels)')
    total_area = pixelareakm2(total)
    print(f'Test {total:.2E} {total_area:.0f}')


if __name__ == '__main__':
    config_name = 'fusionda_extended'
    # train_validation_statistics(config_name)
    # run_train_validation_statistics(config_name)
    # run_test_statistics(config_name)
    plot_train_validation(config_name)
    show_validation_training(config_name)
    # test_statistics(config_name)
    # plot_test(config_name)
    # show_test(config_name)
    # plot_test(config_name)
