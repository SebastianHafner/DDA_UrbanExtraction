import sys
from pathlib import Path
import os
import timeit

import torch
import torch.nn as nn
from torch import optim
from torch.utils import data as torch_data

from tabulate import tabulate
import wandb

from networks.network_loader import create_network, save_checkpoint

from utils.datasets import SelfsupervisedUrbanExtractionDataset
from utils.augmentations import *
from utils.loss import get_criterion
from utils.evaluation import model_evaluation_regression

from experiment_manager.args import default_argument_parser
from experiment_manager.config import config


def run_pretraining(cfg):
    run_config = {
        'CONFIG_NAME': cfg.NAME,
        'device': device,
        'epochs': cfg.TRAINER_PRETRAINING.EPOCHS,
        'learning rate': cfg.TRAINER_PRETRAINING.LR,
        'batch size': cfg.TRAINER_PRETRAINING.BATCH_SIZE,
    }
    table = {'run config name': run_config.keys(),
             ' ': run_config.values(),
             }
    print(tabulate(table, headers='keys', tablefmt="fancy_grid", ))

    net = create_network(cfg)
    net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=float(cfg.TRAINER_PRETRAINING.LR), weight_decay=0.01)

    criterion = get_criterion(cfg.TRAINER_PRETRAINING.LOSS_TYPE)

    # reset the generators
    dataset = SelfsupervisedUrbanExtractionDataset(cfg, 'training')
    print(dataset)

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER_PRETRAINING.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.TRAINER_PRETRAINING.EPOCHS
    save_checkpoints = cfg.SAVE_CHECKPOINTS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        loss_set = []

        for i, batch in enumerate(dataloader):

            net.train()
            optimizer.zero_grad()

            img_optical = batch['x'].to(device)
            img_sar = batch['y'].to(device)

            logits_sar = net(img_optical)
            pred_sar = torch.sigmoid(logits_sar)

            loss = criterion(pred_sar, img_sar)
            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())

            global_step += 1
            epoch_float = global_step / steps_per_epoch
            if global_step % cfg.LOG_FREQ == 0 and not cfg.DEBUG:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # evaluation on sample of training and validation set
                model_evaluation_regression(net, cfg, device, 'training', epoch_float, global_step, max_samples=1_000)
                model_evaluation_regression(net, cfg, device, 'validation', epoch_float, global_step, max_samples=1_000)

                # logging
                time = timeit.default_timer() - start
                wandb.log({
                    'loss': np.mean(loss_set),
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_set = []

            if cfg.DEBUG:
                break
            # end of batch

        if not cfg.DEBUG:
            assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        if epoch in save_checkpoints and not cfg.DEBUG:
            print(f'saving network', flush=True)
            save_checkpoint(net, optimizer, epoch, global_step, cfg)

if __name__ == '__main__':

    args = default_argument_parser().parse_known_args()[0]
    cfg = config.setup(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cudnn.benchmark = True # faster convolutions, but more memory

    print('=== Runnning on device: p', device)

    if not cfg.DEBUG:
        wandb.init(
            name=cfg.NAME,
            project='urban_extraction_selfsupervised',
            tags=['run', 'urban', 'extraction', 'segmentation', ],
        )

    try:
        run_pretraining(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
