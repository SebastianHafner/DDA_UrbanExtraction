import torch
from torch.utils import data as torch_data
import numpy as np
import wandb
from tqdm import tqdm
from utils import datasets, metrics


# specific threshold creates an additional log for that threshold
# can be used to apply best training threshold to validation set
def model_evaluation(net, cfg, device, thresholds: torch.Tensor, run_type: str, epoch: float, step: int,
                     max_samples: int = None, specific_index: int = None):

    thresholds = thresholds.to(device)
    measurer = metrics.MultiThresholdMetric(thresholds)

    def evaluate(y_true, y_pred):
        y_true = y_true.detach()
        y_pred = y_pred.detach()
        measurer.add_sample(y_true, y_pred)

    dataset = datasets.UrbanExtractionDataset(cfg=cfg, dataset=run_type, no_augmentations=True,
                                              include_unlabeled=False)
    inference_loop(net, cfg, device, evaluate, max_samples=max_samples, dataset=dataset)

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

    if specific_index is not None:
        specific_f1 = f1s[specific_index]
        specific_thresh = thresholds[specific_index]
        specific_precision = precisions[specific_index]
        specific_recall = recalls[specific_index]
        if not cfg.DEBUG:
            wandb.log({f'{run_type} specific F1': specific_f1,
                       f'{run_type} specific threshold': specific_thresh,
                       f'{run_type} specific precision': specific_precision,
                       f'{run_type} specific recall': specific_recall,
                       'step': step, 'epoch': epoch,
                       })

    if not cfg.DEBUG:
        wandb.log({f'{run_type} F1': f1,
                   f'{run_type} threshold': best_thresh,
                   f'{run_type} precision': precision,
                   f'{run_type} recall': recall,
                   'step': step, 'epoch': epoch,
                   })

    return argmax_f1.item()


# for regression
def model_evaluation_regression(net, cfg, device, run_type: str, epoch: float, step: int, max_samples: int = None):

    measurer = metrics.RegressionEvaluation()

    def evaluate(y_true, y_pred):
        y_true = y_true.detach()
        y_pred = y_pred.detach()
        measurer.add_sample(y_true, y_pred)

    dataset = datasets.SelfsupervisedUrbanExtractionDataset(cfg=cfg, dataset=run_type, no_augmentations=True)
    inference_loop(net, cfg, device, evaluate, max_samples=max_samples, dataset=dataset)

    print(f'Computing {run_type} RMSE ', end=' ', flush=True)
    rmse = measurer.root_mean_square_error()
    print(f'{rmse.item():.3f}', flush=True)

    if not cfg.DEBUG:
        wandb.log({f'{run_type} RMSE': rmse.item(),
                   'step': step, 'epoch': epoch,
                   })


def model_testing(net, cfg, device, argmax, step, epoch):

    net.eval()

    threshold = argmax / 100

    # loading dataset
    dataset = datasets.SpaceNet7Dataset(cfg)

    y_true_dict = {'total': np.array([])}
    y_pred_dict = {'total': np.array([])}

    for index in tqdm(range(len(dataset))):
        sample = dataset.__getitem__(index)

        with torch.no_grad():
            x = sample['x'].to(device)
            y_true = sample['y'].to(device)
            logits = net(x.unsqueeze(0))
            y_pred = torch.sigmoid(logits) > threshold

            y_true = y_true.detach().cpu().flatten().numpy()
            y_pred = y_pred.detach().cpu().flatten().numpy()

            group_name = sample['group_name']
            if group_name not in y_true_dict.keys():
                y_true_dict[group_name] = y_true
                y_pred_dict[group_name] = y_pred
            else:
                y_true_dict[group_name] = np.concatenate((y_true_dict[group_name], y_true))
                y_pred_dict[group_name] = np.concatenate((y_pred_dict[group_name], y_pred))

            y_true_dict['total'] = np.concatenate((y_true_dict['total'], y_true))
            y_pred_dict['total'] = np.concatenate((y_pred_dict['total'], y_pred))

    def evaluate_group(group_name):
        group_y_true = torch.Tensor(np.array(y_true_dict[group_name]))
        group_y_pred = torch.Tensor(np.array(y_pred_dict[group_name]))
        prec = metrics.precision(group_y_true, group_y_pred, dim=0).item()
        rec = metrics.recall(group_y_true, group_y_pred, dim=0).item()
        f1 = metrics.f1_score(group_y_true, group_y_pred, dim=0).item()

        print(f'{group_name} F1 {f1:.3f} - Precision {prec:.3f} - Recall {rec:.3f}')

        if not cfg.DEBUG:
            wandb.log({f'{group_name} F1': f1,
                       f'{group_name} precision': prec,
                       f'{group_name} recall': rec,
                       'step': step, 'epoch': epoch,
                       })

    for group_index, group_name in dataset.group_names.items():
        evaluate_group(group_name)
    evaluate_group('total')


def model_testing_cucd(net, cfg, device, step, epoch):
    net.eval()

    # loading dataset
    dataset = datasets.CUCDTestingDataset(cfg)

    y_preds, y_trues = [], []

    for index in tqdm(range(len(dataset))):
        sample = dataset.__getitem__(index)

        with torch.no_grad():
            x = sample['x'].to(device)
            y_true = sample['y'].to(device)
            logits = net(x.unsqueeze(0))
            y_pred = torch.sigmoid(logits) > 0.5
            y_true = y_true.detach().cpu().flatten().numpy()
            not_nan = ~np.isnan(y_true)
            y_trues.append(y_true[not_nan])
            y_pred = y_pred.detach().cpu().flatten().numpy()
            y_preds.append(y_pred[not_nan])

    y_trues = torch.Tensor(np.concatenate(y_trues))
    y_preds = torch.Tensor(np.concatenate(y_preds))
    prec = metrics.precision(y_trues, y_preds, dim=0).item()
    rec = metrics.recall(y_trues, y_preds, dim=0).item()
    f1 = metrics.f1_score(y_trues, y_preds, dim=0).item()

    print(f'F1 {f1:.3f} - Precision {prec:.3f} - Recall {rec:.3f}')

    if not cfg.DEBUG:
        wandb.log({f'F1': f1,
                   f'precision': prec,
                   f'recall': rec,
                   'step': step, 'epoch': epoch,
                   })


def inference_loop(net, cfg, device, callback=None, batch_size: int = 1, max_samples: int = None,
                   dataset=None, callback_include_x=False):
    net.to(device)
    net.eval()

    # reset the generators
    num_workers = 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER
    dataloader = torch_data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                       shuffle=True, drop_last=True)
    stop_step = len(dataloader) if max_samples is None else max_samples
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            if step == stop_step:
                break

            imgs = batch['x'].to(device)
            y_label = batch['y'].to(device)

            y_pred = net(imgs)
            y_pred = torch.sigmoid(y_pred)

            if callback:
                if callback_include_x:
                    callback(imgs, y_label, y_pred)
                else:
                    callback(y_label, y_pred)

            if cfg.DEBUG:
                break
