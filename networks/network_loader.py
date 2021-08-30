import torch
from torch import optim

import segmentation_models_pytorch as smp
from networks.unet import UNet, DualStreamUNet
from networks.densefusionnet import DenseFusionNet
from networks.customnets import SimpleNet1
from networks.emanet import EMA
from networks.confidencenet import ConfidenceNet
from networks.original_unet import OriginalUNet
from networks.resnet import ResNet


from pathlib import Path


def get_network(cfg):
    if not cfg.RESUME_CHECKPOINT:
        net = create_network(cfg)
        optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)
    else:
        net, optimizer = load_checkpoint(cfg.RESUME_CHECKPOINT, cfg)
    return net, optimizer


def create_network(cfg):

    architecture = cfg.MODEL.TYPE

    if architecture == 'unet':

        if cfg.MODEL.BACKBONE.ENABLED:
            net = smp.Unet(
                cfg.MODEL.BACKBONE.TYPE,
                encoder_weights=cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS,
                in_channels=cfg.MODEL.IN_CHANNELS,
                classes=cfg.MODEL.OUT_CHANNELS,
                activation=None,
            )
        else:
            net = UNet(cfg)

    elif architecture == 'dualstreamunet':
        net = DualStreamUNet(cfg)

    elif architecture == 'densefusionnet':
        net = DenseFusionNet(cfg)

    elif architecture == 'simplenet1':
        net = SimpleNet1(cfg)

    elif architecture == 'confidencenet':
        net = ConfidenceNet(cfg)

    elif architecture == 'originalunet':
        net = OriginalUNet(cfg)

    else:
        net = UNet(cfg)

    return net


def load_network(cfg, pkl_file: Path):

    net = create_network(cfg)
    state_dict = torch.load(str(pkl_file), map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict)

    return net


def create_ema_network(net, cfg):
    ema_net = EMA(net, decay=cfg.CONSISTENCY_TRAINER.WEIGHT_DECAY)
    return ema_net


def save_checkpoint(network, optimizer, epoch, step, cfg):
    save_file = Path(cfg.OUTPUT_BASE_DIR) / f'{cfg.NAME}_checkpoint{epoch}.pt'
    checkpoint = {
        'step': step,
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_file)


def load_checkpoint(epoch, cfg, device):

    net = create_network(cfg)
    net.to(device)

    save_file = Path(cfg.OUTPUT_BASE_DIR) / f'{cfg.NAME}_checkpoint{epoch}.pt'
    checkpoint = torch.load(save_file, map_location=device)

    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    net.load_state_dict(checkpoint['network'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return net, optimizer, checkpoint['step']
