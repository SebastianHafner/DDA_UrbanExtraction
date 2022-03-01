import torch
from torch.utils import data as torch_data
import numpy as np
import wandb
from tqdm import tqdm
from utils import datasets, metrics


def model_evaluation(net, cfg, device, run_type: str, epoch: float, step: int, max_samples: int = None):
    net.to(device)
    net.eval()

    thresholds = torch.linspace(0.5, 1, 1).to(device)
    measurer = metrics.MultiThresholdMetric(thresholds)

    dataset = datasets.UrbanExtractionDataset(cfg=cfg, dataset=run_type, no_augmentations=True, include_unlabeled=False)

    # reset the generators
    num_workers = 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER
    dataloader = torch_data.DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=True, drop_last=True)

    stop_step = len(dataloader) if max_samples is None else max_samples

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step == stop_step:
                break

            imgs = batch['x'].to(device)
            y_true = batch['y'].to(device)

            y_pred = net(imgs)
            y_pred = torch.sigmoid(y_pred)

            y_true = y_true.detach()
            y_pred = y_pred.detach()
            measurer.add_sample(y_true, y_pred)

            if cfg.DEBUG:
                break

    print(f'Computing {run_type} F1 score ', end=' ', flush=True)

    f1s = measurer.compute_f1()
    precisions, recalls = measurer.precision, measurer.recall

    # best f1 score for passed thresholds
    f1 = f1s.max()
    argmax_f1 = f1s.argmax()

    best_thresh = thresholds[argmax_f1]
    precision = precisions[argmax_f1]
    recall = recalls[argmax_f1]

    print(f'{f1.item():.3f}', flush=True)

    wandb.log({f'{run_type} F1': f1,
               f'{run_type} threshold': best_thresh,
               f'{run_type} precision': precision,
               f'{run_type} recall': recall,
               'step': step, 'epoch': epoch,
               })


def model_testing(net, cfg, device, step, epoch):
    net.to(device)
    net.eval()

    dataset = datasets.SpaceNet7Dataset(cfg)

    y_true_dict = {'total': []}
    y_pred_dict = {'total': []}

    for index in range(len(dataset)):
        sample = dataset.__getitem__(index)

        with torch.no_grad():
            x = sample['x'].to(device)
            y_true = sample['y'].to(device)
            logits = net(x.unsqueeze(0))
            y_pred = torch.sigmoid(logits) > 0.5

            y_true = y_true.detach().cpu().flatten().numpy()
            y_pred = y_pred.detach().cpu().flatten().numpy()

            region = sample['region']
            if region not in y_true_dict.keys():
                y_true_dict[region] = [y_true]
                y_pred_dict[region] = [y_pred]
            else:
                y_true_dict[region].append(y_true)
                y_pred_dict[region].append(y_pred)

            y_true_dict['total'].append(y_true)
            y_pred_dict['total'].append(y_pred)

    def evaluate_region(region_name: str):
        y_true_region = torch.Tensor(np.concatenate(y_true_dict[region_name])).flatten()
        y_pred_region = torch.Tensor(np.concatenate(y_pred_dict[region_name])).flatten()
        prec = metrics.precision(y_true_region, y_pred_region, dim=0).item()
        rec = metrics.recall(y_true_region, y_pred_region, dim=0).item()
        f1 = metrics.f1_score(y_true_region, y_pred_region, dim=0).item()

        wandb.log({f'{region_name} F1': f1,
                   f'{region_name} precision': prec,
                   f'{region_name} recall': rec,
                   'step': step, 'epoch': epoch,
                   })

    for region in dataset.regions.values():
        evaluate_region(region)
    evaluate_region('total')


