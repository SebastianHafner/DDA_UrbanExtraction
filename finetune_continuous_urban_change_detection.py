import sys
from pathlib import Path
import os
from tqdm import tqdm
import wandb

import torch
from torch.utils import data as torch_data

from networks.network_loader import create_network, save_checkpoint, load_checkpoint

from utils import datasets, metrics
from utils.augmentations import *
from utils.loss import get_criterion
from utils.evaluation import model_evaluation, model_testing

from experiment_manager.args import default_argument_parser
from experiment_manager.config import config


def run_finetuning(cfg):

    net, optimizer, global_step = load_checkpoint(cfg.FINETUNING.LOAD_CHECKPOINT, cfg, device)
    consistency_criterion = get_criterion(cfg.CONSISTENCY_TRAINER.CONSISTENCY_LOSS_TYPE)
    sar_criterion = get_criterion(cfg.MODEL.LOSS_TYPE)
    optical_criterion = get_criterion(cfg.MODEL.LOSS_TYPE)
    fusion_criterion = get_criterion(cfg.MODEL.LOSS_TYPE)

    # reset the generators
    dataset = datasets.CUCDFinetuningDataset(cfg, 'spacenet7_s1s2_dataset_v3', 'training')
    print(dataset)

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    start_epoch = epoch_float = cfg.FINETUNING.LOAD_CHECKPOINT
    finetuning_step = 0
    steps_per_epoch = len(dataloader)
    epochs = start_epoch + cfg.FINETUNING.EPOCHS

    # model_finetuning_testing(net, cfg, device, epoch_float + 1, finetuning_step)

    for epoch in range(start_epoch + 1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        supervised_loss_set, consistency_loss_set, loss_set = [], [], []

        for i, batch in enumerate(tqdm(dataloader)):

            net.train()
            optimizer.zero_grad()

            x_fusion = batch['x'].to(device)
            y_gts = batch['y'].to(device)
            is_labeled = batch['is_labeled']
            y_gts = y_gts[is_labeled]

            sar_logits, optical_logits, fusion_logits = net(x_fusion)

            supervised_loss, consistency_loss = None, None

            # supervised loss
            if is_labeled.any():
                sar_loss = sar_criterion(sar_logits[is_labeled], y_gts)
                optical_loss = optical_criterion(optical_logits[is_labeled], y_gts)
                fusion_loss = fusion_criterion(fusion_logits[is_labeled], y_gts)
                supervised_loss = sar_loss + optical_loss + fusion_loss
                supervised_loss_set.append(supervised_loss.item())

            # consistency loss for semi-supervised training
            if not is_labeled.all():
                not_labeled = torch.logical_not(is_labeled)
                sar_probs = torch.sigmoid(sar_logits)
                sar_probs = (sar_probs > 0.5).float() if cfg.CONSISTENCY_TRAINER.APPLY_THRESHOLD else sar_probs
                consistency_loss = consistency_criterion(optical_logits[not_labeled,], sar_probs[not_labeled, ])
                consistency_loss = cfg.CONSISTENCY_TRAINER.LOSS_FACTOR * consistency_loss
                consistency_loss_set.append(consistency_loss.item())

            if supervised_loss is None and consistency_loss is not None:
                loss = consistency_loss
            elif supervised_loss is not None and consistency_loss is not None:
                loss = supervised_loss + consistency_loss
            else:
                loss = supervised_loss

            loss_set.append(loss.item())
            loss.backward()
            optimizer.step()

            global_step += 1
            finetuning_step += 1
            epoch_float = start_epoch + finetuning_step / steps_per_epoch

            if finetuning_step % cfg.FINETUNING.LOG_FREQ == 0 and not cfg.DEBUG:

                wandb.log({
                    'supervised_loss': np.mean(supervised_loss_set),
                    'consistency_loss': np.mean(consistency_loss_set) if consistency_loss_set else 0,
                    'loss_set': np.mean(loss_set),
                    'step': finetuning_step,
                    'epoch': epoch_float,
                })
                supervised_loss_set, consistency_loss_set, loss_set = [], [], []

            if cfg.DEBUG:
                break
            # end of batch

        if not cfg.DEBUG:
            assert (epoch == epoch_float)

        model_finetuning_testing(net, cfg, device, epoch_float, finetuning_step)

        if epoch == epochs and not cfg.DEBUG:
            print(f'saving network', flush=True)
            cfg.NAME = f'{cfg.NAME}_finetuned'
            save_checkpoint(net, optimizer, epoch, global_step, cfg)


def model_finetuning_testing(net, cfg, device, epoch, step):
    net.eval()

    # loading dataset
    dataset = datasets.CUCDFinetuningDataset(cfg, 'spacenet7_s1s2_dataset_v3', 'testing')

    y_trues, y_preds = [], []
    n_y_trues = n_y_preds = 0

    for index in tqdm(range(len(dataset))):
        sample = dataset.__getitem__(index)

        with torch.no_grad():
            x = sample['x'].to(device)
            y_true = sample['y'].to(device)
            logits = net(x.unsqueeze(0))
            y_pred = torch.sigmoid(logits) > 0.5

            y_true = y_true.detach().cpu().flatten().numpy()
            y_pred = y_pred.detach().cpu().flatten().numpy()

            notnan = ~np.isnan(y_true)
            y_true = y_true[notnan]
            y_pred = y_pred[notnan]
            n_y_trues += np.size(y_true)
            n_y_preds += np.size(y_pred)
            assert(n_y_trues == n_y_preds)

            y_trues.append(y_true)
            y_preds.append(y_pred)

    y_trues = torch.Tensor(np.concatenate(y_trues))
    y_preds = torch.Tensor(np.concatenate(y_preds))
    prec = metrics.precision(y_trues, y_preds, dim=0).item()
    rec = metrics.recall(y_trues, y_preds, dim=0).item()
    f1 = metrics.f1_score(y_trues, y_preds, dim=0).item()
    print(f'Epoch {epoch:.1f}: F1 {f1:.3f} - P {prec:.3f} - R {rec:.3f}')

    if not cfg.DEBUG:
        wandb.log({
            'f1_score': f1,
            'precision': prec,
            'recall': rec,
            'step': step,
            'epoch': epoch,
        })


if __name__ == '__main__':

    args = default_argument_parser().parse_known_args()[0]
    cfg = config.setup(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=== Runnning on device: p', device)

    if not cfg.DEBUG:
        wandb.init(
            name=cfg.NAME,
            project='urban_extraction_finetuning',
            tags=['run', 'urban', 'extraction', 'cucd', 'segmentation', ],
        )

    try:
        run_finetuning(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
