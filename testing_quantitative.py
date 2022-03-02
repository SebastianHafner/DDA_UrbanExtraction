from pathlib import Path
import numpy as np
import matplotlib as mpl
from utils import geofiles, parsers, metrics

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


def get_quantitative_data(output_path: str, config_name: str):
    data_file = Path(output_path) / 'testing' / f'probabilities_{config_name}.npy'
    assert (data_file.exists())
    data = np.load(data_file, allow_pickle=True)
    data = dict(data[()])
    return data


# TODO: write this to file
def quantitative_analysis(config_name: str, output_path: str):
    threshold = 0.2 if config_name == 'ghs' else 0.5
    data = get_quantitative_data(output_path, config_name)
    output = {}
    for metric in ['f1_score', 'precision', 'recall', 'iou']:
        print(metric)
        region_values = []
        for region_name, region in data.items():

            y_true = np.concatenate([site['y_true'] for site in region], axis=0)
            y_prob = np.concatenate([site['y_prob'] for site in region], axis=0)

            if metric == 'f1_score':
                value = metrics.f1_score_from_prob(y_prob, y_true, threshold)
            elif metric == 'precision':
                value = metrics.precision_from_prob(y_prob, y_true, threshold)
            elif metric == 'recall':
                value = metrics.recall_from_prob(y_prob, y_true, threshold)
            else:
                value = metrics.iou_from_prob(y_prob, y_true, threshold)
            region_values.append(value)

        output[f'regional_{metric}'] = {
            'min': np.min(region_values),
            'max': np.max(region_values),
            'mean': np.mean(region_values),
            'std': np.std(region_values)
        }

    y_true = np.concatenate([site['y_true'] for region in data.values() for site in region], axis=0)
    y_prob = np.concatenate([site['y_prob'] for region in data.values() for site in region], axis=0)
    output['total_f1_score'] = metrics.f1_score_from_prob(y_prob, y_true, threshold)
    output['total_precision'] = metrics.precision_from_prob(y_prob, y_true, threshold)
    output['total_recall'] = metrics.recall_from_prob(y_prob, y_true, threshold)
    output['total_iou'] = metrics.iou_from_prob(y_prob, y_true, threshold)

    out_file = Path(output_path) / 'testing' / f'quantitative_results_{config_name}.json'
    geofiles.write_json(out_file, output)


if __name__ == '__main__':
    args = parsers.testing_inference_argument_parser().parse_known_args()[0]
    quantitative_analysis(args.config_file, args.output_dir)
