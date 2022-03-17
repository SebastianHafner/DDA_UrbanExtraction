from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from utils import metrics, geofiles, experiment_manager, networks, datasets, parsers


def get_spacenet7_aoi_ids(dataset_path: str) -> list:
    file = Path(dataset_path) / 'spacenet7' / 'spacenet7_regions.json'
    metadata_regions = geofiles.load_json(file)
    aoi_ids = metadata_regions['data'].keys()
    return sorted(aoi_ids)


def get_region_name(dataset_path: str, aoi_id: str) -> str:
    file = Path(dataset_path) / 'spacenet7' / 'spacenet7_regions.json'
    metadata_regions = geofiles.load_json(file)
    region_index = metadata_regions['data'][aoi_id]
    region_name = metadata_regions['regions'][str(region_index)]
    return region_name


def get_ghs_threshold(dataset_path: str, aoi_id: str) -> float:
    file = Path(dataset_path) / 'spacenet7' / 'ghs_thresholds.json'
    ghs_thresholds = geofiles.load_json(file)
    threshold = float(ghs_thresholds[aoi_id])
    return threshold


def run_inference_ours(cfg: experiment_manager):
    # loading config and network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _, _ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()

    dataset = datasets.SpaceNet7Dataset(cfg)

    with torch.no_grad():
        for index in tqdm(range(len(dataset))):
            test_site = dataset.__getitem__(index)
            aoi_id = test_site['aoi_id']
            img = test_site['x'].to(device)
            y_prob = net(img.unsqueeze(0))
            y_prob = torch.sigmoid(y_prob).squeeze().cpu().numpy()

            output_folder = Path(cfg.PATHS.DATASET) / 'spacenet7' / cfg.NAME
            output_folder.mkdir(exist_ok=True)
            output_file = output_folder / f'{cfg.NAME}_{aoi_id}.tif'
            transform, crs = test_site['transform'], test_site['crs']
            geofiles.write_tif(output_file, y_prob, transform, crs)


def run_quantitative_inference_ours(cfg: experiment_manager.CfgNode, threshold: float = 0.5):
    # loading config and network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _, _ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()

    dataset = datasets.SpaceNet7Dataset(cfg)

    data = {}
    y_preds = []
    y_trues = []
    with torch.no_grad():
        for index in tqdm(range(len(dataset))):
            test_site = dataset.__getitem__(index)
            img = test_site['x'].to(device)
            y_prob = net(img.unsqueeze(0))
            y_prob = torch.sigmoid(y_prob).flatten().cpu().numpy()
            y_true = test_site['y'].flatten().cpu().numpy()
            y_trues.append(y_true)

            region_name = test_site['region']
            if region_name not in data.keys():
                data[region_name] = []

            y_pred = (y_prob > threshold).astype(np.float32)
            y_preds.append(y_pred)

            f1 = metrics.f1_score_from_prob(y_prob, y_true, threshold)
            p = metrics.precision_from_prob(y_prob, y_true, threshold)
            r = metrics.recall_from_prob(y_prob, y_true, threshold)
            iou = metrics.iou_from_prob(y_prob, y_true, threshold)

            site_data = {
                'aoi_id': test_site['aoi_id'],
                'y_prob': y_prob,
                'y_pred': y_pred,
                'y_true': y_true,
                'threshold': threshold,
                'f1_score': f1,
                'precision': p,
                'recall': r,
                'iou': iou,
            }

            data[region_name].append(site_data)

    y_preds = np.concatenate(y_preds)
    y_trues = np.concatenate(y_trues)
    data['total_f1_score'] = metrics.f1_score_from_prob(y_preds, y_trues, 0.5)
    data['total_precision'] = metrics.precision_from_prob(y_preds, y_trues, 0.5)
    data['total_recall'] = metrics.recall_from_prob(y_preds, y_trues, 0.5)
    data['total_iou'] = metrics.iou_from_prob(y_preds, y_trues, 0.5)

    output_file = Path(cfg.PATHS.OUTPUT) / 'testing' / f'probabilities_{cfg.NAME}.npy'
    output_file.parent.mkdir(exist_ok=True)
    np.save(output_file, data)


def run_quantitative_inference_sota(dataset_path: str, output_path: str, sota_name: str):
    data = {}
    aoi_ids = get_spacenet7_aoi_ids(dataset_path)
    y_preds = []
    y_trues = []
    for aoi_id in aoi_ids:

        file = Path(dataset_path) / 'spacenet7' / sota_name / f'{sota_name}_{aoi_id}.tif'
        sota, *_ = geofiles.read_tif(file)
        sota = sota.flatten().astype(np.float32)
        if sota_name == 'wsf2019':
            sota = sota / 255

        # ground truth
        file = Path(dataset_path) / 'spacenet7' / 'buildings' / f'buildings_{aoi_id}.tif'
        y_true, *_ = geofiles.read_tif(file)
        y_true = (y_true.flatten() > 0).astype(np.float32)
        y_trues.append(y_true)

        region_name = get_region_name(dataset_path, aoi_id)
        if region_name not in data.keys():
            data[region_name] = []

        threshold = get_ghs_threshold(dataset_path, aoi_id) if sota_name == 'ghs' else 0.5
        y_pred = (sota > threshold).astype(np.float32)
        y_preds.append(y_pred)
        f1 = metrics.f1_score_from_prob(sota, y_true, threshold)
        p = metrics.precision_from_prob(sota, y_true, threshold)
        r = metrics.recall_from_prob(sota, y_true, threshold)
        iou = metrics.iou_from_prob(sota, y_true, threshold)

        site_data = {
            'aoi_id': aoi_id,
            'y_prob': sota,
            'y_pred': y_pred,
            'y_true': y_true,
            'threshold': threshold,
            'f1_score': f1,
            'precision': p,
            'recall': r,
            'iou': iou,
        }

        data[region_name].append(site_data)

    y_preds = np.concatenate(y_preds)
    y_trues = np.concatenate(y_trues)
    data['total_f1_score'] = metrics.f1_score_from_prob(y_preds, y_trues, 0.5)
    data['total_precision'] = metrics.precision_from_prob(y_preds, y_trues, 0.5)
    data['total_recall'] = metrics.recall_from_prob(y_preds, y_trues, 0.5)
    data['total_iou'] = metrics.iou_from_prob(y_preds, y_trues, 0.5)

    output_file = Path(output_path) / 'testing' / f'probabilities_{sota_name}.npy'
    output_file.parent.mkdir(exist_ok=True)
    np.save(output_file, data)


if __name__ == '__main__':
    args = parsers.testing_inference_argument_parser().parse_known_args()[0]

    if args.config_file == 'ghs' or args.config_file == 'wsf2019':
        run_quantitative_inference_sota(args.dataset_dir, args.output_dir, args.config_file)
    else:
        cfg = experiment_manager.setup_cfg(args)
        run_inference_ours(cfg)
        run_quantitative_inference_ours(cfg)
