import sys
from pathlib import Path
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

from tabulate import tabulate
import wandb

from networks.network_loader import create_network, load_network

from utils.datasets import STUrbanExtractionDataset
from utils.augmentations import *
from utils.evaluation import model_evaluation
from utils.loss import get_criterion

from experiment_manager.args import default_argument_parser
from experiment_manager.config import new_config

from tqdm import tqdm


def train_sar_teacher(cfg, sar_cfg):
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

    net = create_network(cfg)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)
    supervised_criterion = get_criterion(cfg.MODEL.LOSS_TYPE)
    net.to(device)

    sar_net_file = Path(sar_cfg.OUTPUT_BASE_DIR) / f'{sar_cfg.NAME}_{sar_cfg.INFERENCE.CHECKPOINT}.pkl'
    sar_net = load_network(sar_cfg, sar_net_file)
    sar_net.to(device)
    sar_net.eval()
    consistency_criterion = get_criterion(cfg.CONSISTENCY_TRAINER.CONSISTENCY_LOSS_TYPE)

    dataset = STUrbanExtractionDataset(cfg=cfg, sar_cfg=sar_cfg, run_type='training')
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
    save_checkpoints = cfg.SAVE_CHECKPOINTS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = 0

    for epoch in range(epochs):
        print(f'Starting epoch {epoch + 1}/{epochs}.')

        start = timeit.default_timer()
        loss_set, supervised_loss_set, consistency_loss_set = [], [], []
        n_labeled, n_notlabeled = 0, 0

        net.train()

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            x = batch['x'].to(device)
            x_sar = batch['x_sar'].to(device)
            y_gts = batch['y'].to(device)
            is_labeled = batch['is_labeled']

            logits = net(x)

            supervised_loss, consistency_loss = None, None

            if is_labeled.any():
                supervised_loss = supervised_criterion(logits[is_labeled, ], y_gts[is_labeled])
                supervised_loss_set.append(supervised_loss.item())
                n_labeled += torch.sum(is_labeled).item()

            if not is_labeled.all():
                not_labeled = torch.logical_not(is_labeled)
                with torch.no_grad():
                    probs_sar = torch.sigmoid(sar_net(x_sar[not_labeled, ]))
                    if cfg.CONSISTENCY_TRAINER.APPLY_THRESHOLD:
                        output_sar = (probs_sar > sar_cfg.INFERENCE.THRESHOLDS.VALIDATION).float()
                    else:
                        output_sar = probs_sar

                consistency_loss = consistency_criterion(logits[not_labeled, ], output_sar)
                consistency_loss_set.append(consistency_loss.item())
                n_notlabeled += torch.sum(not_labeled).item()

            if supervised_loss is None and consistency_loss is not None:
                loss = cfg.CONSISTENCY_TRAINER.LOSS_FACTOR * consistency_loss
            elif supervised_loss is not None and consistency_loss is not None:
                loss = supervised_loss + cfg.CONSISTENCY_TRAINER.LOSS_FACTOR * consistency_loss
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
                thresholds = torch.linspace(0, 1, 101)
                train_argmaxF1 = model_evaluation(net, cfg, device, thresholds, 'training', epoch_float,
                                                  global_step, max_samples=1_000)
                _ = model_evaluation(net, cfg, device, thresholds, 'validation', epoch_float, global_step,
                                     specific_index=train_argmaxF1, max_samples=1_000)

                # logging
                time = timeit.default_timer() - start
                labeled_percentage = n_labeled / (n_labeled + n_notlabeled) * 100
                wandb.log({
                    'loss': np.mean(loss_set),
                    'supervised_loss': 0 if len(supervised_loss_set) == 0 else np.mean(supervised_loss_set),
                    'consistency_loss': 0 if len(consistency_loss_set) == 0 else np.mean(consistency_loss_set),
                    'labeled_percentage': labeled_percentage,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })

                # resetting stuff
                start = timeit.default_timer()
                loss_set, supervised_loss_set, consistency_loss_set = [], [], []
                n_labeled, n_notlabeled = 0, 0

            if cfg.DEBUG:
                break
            # end of batch

        if epoch in save_checkpoints:
            print(f'saving network', flush=True)
            net_file = Path(cfg.OUTPUT_BASE_DIR) / f'{cfg.NAME}_{epoch}.pkl'
            torch.save(net.state_dict(), net_file)


def setup(args):
    cfg = new_config()
    cfg.merge_from_file(f'configs/{args.config_file}.yaml')
    cfg.merge_from_list(args.opts)
    cfg.NAME = args.config_file

    sar_cfg = new_config()
    sar_cfg.merge_from_file(f'configs/{args.sar_config_file}.yaml')
    sar_cfg.merge_from_list(args.opts)
    sar_cfg.NAME = args.sar_config_file

    return cfg, sar_cfg


if __name__ == '__main__':

    args = default_argument_parser().parse_known_args()[0]
    cfg, sar_cfg = setup(args)

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
        train_sar_teacher(cfg, sar_cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
