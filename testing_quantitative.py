from pathlib import Path
import numpy as np
import matplotlib as mpl
from utils import geofiles, parsers, metrics


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


def get_quantitative_data(output_path: str, config_name: str):
    data_file = Path(output_path) / 'testing' / f'probabilities_{config_name}.npy'
    assert (data_file.exists())
    data = np.load(data_file, allow_pickle=True)
    data = dict(data[()])
    return data


def quantitative_analysis(config_name: str, dataset_path: str, output_path: str):
    metric_names = ['f1_score', 'precision', 'recall', 'iou', 'kappa']
    regions = get_region_names(dataset_path)
    data = get_quantitative_data(output_path, config_name)
    output = {}
    for metric in metric_names:
        values = []
        for region in regions:
            y_true = np.concatenate([site['y_true'] for site in data[region]], axis=0)
            y_pred = np.concatenate([site['y_pred'] for site in data[region]], axis=0)
            if metric == 'f1_score':
                value = metrics.f1_score_from_prob(y_pred, y_true)
            elif metric == 'precision':
                value = metrics.precision_from_prob(y_pred, y_true)
            elif metric == 'recall':
                value = metrics.recall_from_prob(y_pred, y_true)
            elif metric == 'iou':
                value = metrics.iou_from_prob(y_pred, y_true)
            else:
                value = metrics.kappa_from_prob(y_pred, y_true)
            values.append(value)
        output[metric] = {
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values),
            'std': np.std(values)
        }
    for metric in metric_names:
        output[f'total_{metric}'] = data[f'total_{metric}']

    out_file = Path(output_path) / 'testing' / f'quantitative_results_{config_name}.json'
    geofiles.write_json(out_file, output)


if __name__ == '__main__':
    args = parsers.testing_inference_argument_parser().parse_known_args()[0]
    quantitative_analysis(args.config_file, args.dataset_dir, args.output_dir)
