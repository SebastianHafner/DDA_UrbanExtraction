from utils.visualization import *
from pathlib import Path
import numpy as np
import utm
import json
import torch
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from networks.network_loader import load_network
from utils.dataloader import SpaceNet7Dataset
from experiment_manager.config import config
from utils.metrics import *
from utils.geotiff import *
from scipy.stats import gaussian_kde


DATASET_PATH = Path('/storage/shafner/urban_extraction/urban_extraction')
CONFIG_PATH = Path('/home/shafner/urban_dl/configs')
NETWORK_PATH = Path('/storage/shafner/urban_extraction/networks')
SN7_PATH = Path('/storage/shafner/urban_extraction/spacenet7')

GROUPS = [(1, 'NA_AU', '#63cd93'), (2, 'SA', '#f0828f'), (3, 'EU', '#6faec9'), (4, 'SSA', '#5f4ad9'),
          (5, 'NAF_ME', '#8dee47'), (6, 'AS', '#d9b657'), ('total', 'Total', '#ffffff')]


def polygonareanp(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def centroidnp(arr: np.ndarray) -> np.ndarray:
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x / length, sum_y / length


def centroid(feature: dict) -> tuple:
    coords = feature['geometry']['coordinates']
    # TODO: solve for polygons that have list of coordinates (i.e. inner polygon)
    coords = np.array(coords[0])
    c = centroidnp(coords)
    lon, lat = c
    return lat, lon


def unpack_transform(arr, transform):

    # TODO: look for a nicer solution here
    if len(arr.shape) == 2:
        y_pixels, x_pixels = arr.shape
    else:
        y_pixels, x_pixels, *_ = arr.shape

    x_pixel_spacing = transform[0]
    x_min = transform[2]
    x_max = x_min + x_pixels * x_pixel_spacing

    y_pixel_spacing = transform[4]
    y_max = transform[5]
    y_min = y_pixels * y_pixel_spacing

    return x_min, x_max, x_pixel_spacing, y_min, y_max, y_pixel_spacing


def is_out_of_bounds(img: np.ndarray, transform, coords) -> bool:
    x_coord, y_coord = coords
    x_min, x_max, _, y_min, y_max, _ = unpack_transform(img, transform)
    if x_coord < x_min or x_coord > x_max:
        return True
    if y_coord < y_min or y_coord > y_max:
        return True
    return False


def is_valid_footprint(footprint):
    if footprint['geometry'] is None:
        return False
    elif footprint['geometry']['type'] != 'Polygon':
        return False
    else:
        return True


def value_at_coords(img: np.ndarray, transform, coords) -> float:
    x_coord, y_coord = coords
    x_min, _, x_pixel_spacing, _, y_max, y_pixel_spacing = unpack_transform(img, transform)
    x_index = int((x_coord - x_min) // x_pixel_spacing)
    y_index = int((y_coord - y_max) // y_pixel_spacing)
    value = img[y_index, x_index, ]
    return value


def load_building_footprints(aoi_id: str, year: int, month: int, wgs84: bool = True):
    file_name = f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.geojson'
    label_folder = 'labels' if wgs84 else 'labels_match'
    label_file = SN7_PATH / 'train' / aoi_id / label_folder / file_name
    label = load_json(label_file)
    features = label['features']
    return features


def run_building_size_experiment(config_name: str, checkpoint: int):
    cfg_file = CONFIG_PATH / f'{config_name}.yaml'
    cfg = config.load_cfg(cfg_file)

    # loading dataset
    dataset = SpaceNet7Dataset(cfg)

    # loading network
    net_file = NETWORK_PATH / f'{config_name}_{checkpoint}.pkl'
    net = load_network(cfg, net_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    output_data = []

    for index in range(len(dataset)):
        sample = dataset.__getitem__(index)
        aoi_id = sample['aoi_id']
        print(f'processing {aoi_id}')

        x = sample['x'].to(device)
        y_true = sample['y'].to(device)
        transform, crs = sample['transform'], sample['crs']

        with torch.no_grad():
            logits = net(x.unsqueeze(0))
            y_prob = torch.sigmoid(logits)
            y_true = torch.squeeze(y_true)
            y_prob = torch.squeeze(y_prob)

            y_true = y_true.detach().cpu().numpy()
            y_prob = y_prob.detach().cpu().numpy()

        footprints = load_building_footprints(sample['aoi_id'], sample['year'], sample['month'], wgs84=False)
        areas = [footprint['properties']['area'] for footprint in footprints]
        footprints = load_building_footprints(sample['aoi_id'], sample['year'], sample['month'])

        for i, (footprint, area) in enumerate(zip(footprints, areas)):
            if is_valid_footprint(footprint):
                building_centroid = centroid(footprint)
                easting, northing, zone_number, zone_letter = utm.from_latlon(*building_centroid)

                # check for out of bounds
                if not is_out_of_bounds(y_prob, transform, (easting, northing)):
                    prob = value_at_coords(y_prob, transform, (easting, northing))
                    output_data.append((float(area), float(prob)))

    output_file = DATASET_PATH.parent / 'building_size_experiment' / f'data_{config_name}.json'
    with open(str(output_file), 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)


def plot_building_size_experiment(config_name: str):
    file = DATASET_PATH.parent / 'building_size_experiment' / f'data_{config_name}.json'
    data = load_json(file)
    areas = np.array([d[0] for d in data])
    prob = np.array([d[1] for d in data])
    print(areas.shape)
    max_index = 100
    x = areas[:max_index]
    y = prob[:max_index]

    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, c=z)
    # ax.scatter(x, y, c=list(z), s=100, edgecolor='')

    ax.set_xlim((0, 10000))
    plt.show()


def bucketize(config_name: str, bucket_size: int, max_size: int = 10_000):
    # extracting threshold for detection rate
    cfg_file = CONFIG_PATH / f'{config_name}.yaml'
    cfg = config.load_cfg(cfg_file)
    threshold = cfg.THRESH

    def was_detected(prob):
        return True if prob > threshold else False

    file = DATASET_PATH.parent / 'building_size_experiment' / f'data_{config_name}.json'
    data = load_json(file)
    data_bucketized = []

    # TODO: include everything larger than max in last bucket
    bucket_starts = np.arange(0, max_size, bucket_size)
    for start in bucket_starts:
        data_bucket = [d for d in data if start < d[0] < start + bucket_size]
        n = len(data_bucket)
        detected = [was_detected(d[1]) for d in data_bucket]

        bucket_stats = {
            'area': [d[0] for d in data_bucket],
            'prob': [d[1] for d in data_bucket],
            'detected': detected,
            'detection_rate': sum(detected) / n,
            'n': n
        }

        data_bucketized.append(bucket_stats)

    return data_bucketized


def boxplots(config_name):

    # Random test data
    np.random.seed(19680801)
    all_data = [np.random.normal(0, std, size=100) for std in range(1, 4)]
    labels = ['x1', 'x2', 'x3']

    fig, ax = plt.subplots(figsize=(4, 4))

    # notch shape box plot
    bplot = ax.boxplot(all_data,
                         notch=True,  # notch shape
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks
    ax.set_title('Notched box plot')

    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    # adding horizontal grid lines
    ax.yaxis.grid(True)
    ax.set_xlabel('Three separate samples')
    ax.set_ylabel('Observed values')

    plt.show()


def line_plot(config_name: str):

    data_bucketized = bucketize(config_name, bucket_size=10, max_size=1_000)
    n_buckets = len(data_bucketized)

    detection_rate = [bucket['detection_rate'] for bucket in data_bucketized]
    n_buildings = [bucket['n'] for bucket in data_bucketized]
    x = np.arange(n_buckets)

    fig, ax1 = plt.subplots(figsize=(10, 4))

    color = 'tab:red'
    ax1.set_xlabel('Building Size')
    ax1.set_ylabel('Detection Rate', color=color)
    ax1.plot(x, detection_rate, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim((0, 1))

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('n Buildings', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, n_buildings, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.show()


if __name__ == '__main__':
    config_name = 'baseline_sar'
    checkpoint = 100
    # run_building_size_experiment(config_name, checkpoint)
    # plot_building_size_experiment(config_name)
    line_plot(config_name)
