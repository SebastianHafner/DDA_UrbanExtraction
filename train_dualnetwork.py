import sys
from pathlib import Path
import os
import timeit

import torch
import torch.nn as nn
from torch.utils import data as torch_data
from torch import optim

from tabulate import tabulate
import wandb

from networks.network_loader import create_network, save_checkpoint, load_checkpoint

from utils.datasets import UrbanExtractionDataset
from utils.augmentations import *
from utils.loss import get_criterion
from utils.evaluation import model_evaluation, model_testing

from experiment_manager.args import default_argument_parser
from experiment_manager.config import config


def run_training(cfg):
    run_config = {
        'CONFIG_NAME': cfg.NAME,
        'device': device,
        'epochs': cfg.TRAINER.EPOCHS,
        'learning rate': cfg.TRAINER.LR,
        'batch size': cfg.TRAINER.BATCH_SIZE,
    }
    table = {'run config name': run_config.keys(),
             ' ': run_config.values(),
             }
    print(tabulate(table, headers='keys', tablefmt="fancy_grid", ))

    if not cfg.RESUME_CHECKPOINT:
        net = create_network(cfg)
        net.to(device)
        optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)
        global_step = 0
    else:
        net, optimizer, global_step = load_checkpoint(cfg.RESUME_CHECKPOINT, cfg, device)

    sar_criterion = get_criterion(cfg.MODEL.LOSS_TYPE)
    optical_criterion = get_criterion(cfg.MODEL.LOSS_TYPE)
    fusion_criterion = get_criterion(cfg.MODEL.LOSS_TYPE)
    consistency_criterion = get_criterion(cfg.CONSISTENCY_TRAINER.CONSISTENCY_LOSS_TYPE)

    # reset the generators
    dataset = UrbanExtractionDataset(cfg=cfg, dataset='training')
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
    epochs = cfg.TRAINER.EPOCHS
    start_epoch = epoch_float = cfg.RESUME_CHECKPOINT
    save_checkpoints = cfg.SAVE_CHECKPOINTS
    steps_per_epoch = len(dataloader)

    # for loggingfusi
    thresholds = torch.linspace(0, 1, 101)

    for epoch in range(start_epoch + 1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        sar_loss_set, optical_loss_set, fusion_loss_set = [], [], []
        supervised_loss_set, consistency_loss_set, loss_set = [], [], []
        n_labeled, n_notlabeled = 0, 0

        for i, batch in enumerate(dataloader):

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
                sar_loss_set.append(sar_loss.item())

                optical_loss = optical_criterion(optical_logits[is_labeled], y_gts)
                optical_loss_set.append(optical_loss.item())

                fusion_loss = fusion_criterion(fusion_logits[is_labeled], y_gts)
                fusion_loss_set.append(fusion_loss.item())
                n_labeled += torch.sum(is_labeled).item()

                supervised_loss = sar_loss + optical_loss + fusion_loss
                supervised_loss_set.append(supervised_loss.item())

            # consistency loss for semi-supervised training
            if not is_labeled.all():
                not_labeled = torch.logical_not(is_labeled)
                n_notlabeled += torch.sum(not_labeled).item()

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
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0 and not cfg.DEBUG:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # evaluation on sample of training and validation set
                train_argmaxF1 = model_evaluation(net, cfg, device, thresholds, 'training', epoch_float, global_step,
                                                  max_samples=1_000)
                _ = model_evaluation(net, cfg, device, thresholds, 'validation', epoch_float, global_step,
                                     specific_index=train_argmaxF1, max_samples=1_000)

                # logging
                time = timeit.default_timer() - start
                labeled_percentage = n_labeled / (n_labeled + n_notlabeled) * 100
                wandb.log({
                    'sar_loss': np.mean(sar_loss_set),
                    'optical_loss': np.mean(optical_loss_set),
                    'fusion_loss': np.mean(fusion_loss_set),
                    'supervised_loss': np.mean(supervised_loss_set),
                    'consistency_loss': np.mean(consistency_loss_set) if consistency_loss_set else 0,
                    'loss_set': np.mean(loss_set),
                    'labeled_percentage': labeled_percentage,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                sar_loss_set, optical_loss_set, fusion_loss_set = [], [], []
                supervised_loss_set, consistency_loss_set, loss_set = [], [], []
                n_labeled, n_notlabeled = 0, 0

            if cfg.DEBUG:
                break
            # end of batch

        if not cfg.DEBUG:
            assert (epoch == epoch_float)
        if epoch in save_checkpoints and not cfg.DEBUG:
            print(f'saving network', flush=True)
            save_checkpoint(net, optimizer, epoch, global_step, cfg)

            # logs to load network
            train_argmaxF1 = model_evaluation(net, cfg, device, thresholds, 'training', epoch_float, global_step)
            validation_argmaxF1 = model_evaluation(net, cfg, device, thresholds, 'validation', epoch_float, global_step,
                                                   specific_index=train_argmaxF1)
            wandb.log({
                'net_checkpoint': epoch,
                'checkpoint_step': global_step,
                'train_threshold': train_argmaxF1 / 100,
                'validation_threshold': validation_argmaxF1 / 100
            })
            model_testing(net, cfg, device, 50, global_step, epoch_float)


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
            project='urban_extraction',
            tags=['run', 'urban', 'extraction', 'segmentation', ],
        )

    try:
        run_training(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)