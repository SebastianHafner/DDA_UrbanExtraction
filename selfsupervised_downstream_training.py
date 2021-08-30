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

from networks.network_loader import create_network, save_checkpoint, load_checkpoint

from utils.datasets import UrbanExtractionDataset
from utils.augmentations import *
from utils.loss import get_criterion
from utils.evaluation import model_evaluation, model_testing

from experiment_manager.args import default_argument_parser
from experiment_manager.config import config


def run_downstream_training(cfg):

    run_config = {
        'CONFIG_NAME': cfg.NAME,
        'device': device,
        'epochs': cfg.TRAINER_DOWNSTREAM.EPOCHS,
        'learning rate': cfg.TRAINER_DOWNSTREAM.LR,
        'batch size': cfg.TRAINER_DOWNSTREAM.BATCH_SIZE,
    }
    table = {'run config name': run_config.keys(),
             ' ': run_config.values(),
             }
    print(tabulate(table, headers='keys', tablefmt="fancy_grid", ))

    # loading config and network
    net_pretrained, _, _ = load_checkpoint(cfg.CHECKPOINTS.DOWNSTREAM, cfg, device)

    # modifying config for downstream task
    cfg.NAME = f'{cfg.NAME}_downstream'
    cfg.LOG_FREQ = cfg.LOG_FREQ_DOWNSTREAM
    cfg.MODEL.OUT_CHANNELS = 1

    if cfg.TRAINER_DOWNSTREAM.FREEZE_ENCODER:
        for i, params_pretrained in enumerate(net_pretrained.parameters()):
            if i < 26:
                params_pretrained.requires_grad = False
                # print(f'{i}: {params_pretrained.shape}')

    net = create_network(cfg).to(device)
    # TODO: make reinitialization of decoder work
    if cfg.TRAINER_DOWNSTREAM.REINITIALIZE_DECODER:
        for i, params_reinitialized in enumerate(net.parameters()):
            if i >= 26:
                net_pretrained.parameters()[i] = params_reinitialized
    net_pretrained.outc = net.outc
    net = None

    optimizer = optim.AdamW(net_pretrained.parameters(), lr=float(cfg.TRAINER_DOWNSTREAM.LR), weight_decay=0.01)

    criterion = get_criterion(cfg.TRAINER_DOWNSTREAM.LOSS_TYPE)

    # reset the generators
    dataset = UrbanExtractionDataset(cfg=cfg, dataset='training')
    print(dataset)

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER_DOWNSTREAM.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle':cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.TRAINER_DOWNSTREAM.EPOCHS
    save_checkpoints = cfg.SAVE_CHECKPOINTS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    # for logging
    thresholds = torch.linspace(0, 1, 101)

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        loss_set = []

        for i, batch in enumerate(dataloader):

            net_pretrained.train()
            optimizer.zero_grad()

            x = batch['x'].to(device)
            y_gts = batch['y'].to(device)

            y_pred = net_pretrained(x)

            loss = criterion(y_pred, y_gts)
            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0 and not cfg.DEBUG:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # evaluation on sample of training and validation set
                train_argmaxF1 = model_evaluation(net_pretrained, cfg, device, thresholds, 'training', epoch_float, global_step,
                                                  max_samples=1_000)
                _ = model_evaluation(net_pretrained, cfg, device, thresholds, 'validation', epoch_float, global_step,
                                     specific_index=train_argmaxF1, max_samples=1_000)

                # logging
                time = timeit.default_timer() - start
                wandb.log({
                    'loss': np.mean(loss_set),
                    'labeled_percentage': 100,
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
            assert(epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        if epoch in save_checkpoints and not cfg.DEBUG:
            print(f'saving network', flush=True)
            save_checkpoint(net_pretrained, optimizer, epoch, global_step, cfg)
            # logs to load network
            train_argmaxF1 = model_evaluation(net_pretrained, cfg, device, thresholds, 'training', epoch_float, global_step)
            validation_argmaxF1 = model_evaluation(net_pretrained, cfg, device, thresholds, 'validation', epoch_float,
                                                   global_step, specific_index=train_argmaxF1)
            wandb.log({
                'net_checkpoint': epoch,
                'checkpoint_step': global_step,
                'train_threshold': train_argmaxF1 / 100,
                'validation_threshold': validation_argmaxF1 / 100
            })
            model_testing(net_pretrained, cfg, device, 50, global_step, epoch_float)


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
            name=f'{cfg.NAME}_downstream',
            project='urban_extraction_selfsupervised',
            tags=['run', 'urban', 'extraction', 'segmentation', ],
        )

    try:
        run_downstream_training(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
