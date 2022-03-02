import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from pathlib import Path

from utils import experiment_manager


def create_network(cfg):
    return DualStreamUNet(cfg) if cfg.MODEL.TYPE == 'dualstreamunet' else UNet(cfg)


def save_checkpoint(network, optimizer, epoch: int, step: int, cfg: experiment_manager.CfgNode):
    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_checkpoint{epoch}.pt'
    checkpoint = {
        'step': step,
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_file)


def load_checkpoint(epoch: int, cfg: experiment_manager.CfgNode, device):
    net = create_network(cfg)
    net.to(device)

    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_checkpoint{epoch}.pt'
    checkpoint = torch.load(save_file, map_location=device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    net.load_state_dict(checkpoint['network'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return net, optimizer, checkpoint['step']


class UNet(nn.Module):
    def __init__(self, cfg, n_channels=None, n_classes=None, topology=None, enable_outc=True):

        self._cfg = cfg

        n_channels = cfg.MODEL.IN_CHANNELS if n_channels is None else n_channels
        n_classes = cfg.MODEL.OUT_CHANNELS if n_classes is None else n_classes
        topology = cfg.MODEL.TOPOLOGY if topology is None else topology

        super(UNet, self).__init__()

        first_chan = topology[0]
        self.inc = InConv(n_channels, first_chan, DoubleConv)
        self.enable_outc = enable_outc
        self.outc = OutConv(first_chan, n_classes)

        # Variable scale
        down_topo = topology
        down_dict = OrderedDict()
        n_layers = len(down_topo)
        up_topo = [first_chan]  # topography upwards
        up_dict = OrderedDict()

        # Downward layers
        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx + 1] if is_not_last_layer else down_topo[idx]  # last layer

            layer = Down(in_dim, out_dim, DoubleConv)

            print(f'down{idx + 1}: in {in_dim}, out {out_dim}')
            down_dict[f'down{idx + 1}'] = layer
            up_topo.append(out_dim)
        self.down_seq = nn.ModuleDict(down_dict)

        # Upward layers
        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]

            layer = Up(in_dim, out_dim, DoubleConv)

            print(f'up{idx + 1}: in {in_dim}, out {out_dim}')
            up_dict[f'up{idx + 1}'] = layer

        self.up_seq = nn.ModuleDict(up_dict)

    def forward(self, x1, x2=None):
        x = x1 if x2 is None else torch.cat((x1, x2), 1)

        x1 = self.inc(x)

        inputs = [x1]
        # Downward U:
        for layer in self.down_seq.values():
            out = layer(inputs[-1])
            inputs.append(out)

        # Upward U:
        inputs.reverse()
        x1 = inputs.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = inputs[idx]
            x1 = layer(x1, x2)  # x1 for next up layer

        out = self.outc(x1) if self.enable_outc else x1

        return out


class DualStreamUNet(nn.Module):

    def __init__(self, cfg):
        super(DualStreamUNet, self).__init__()
        self._cfg = cfg
        out = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY
        out_dim = topology[0]

        # sentinel-1 sar unet stream
        sar_in = len(cfg.DATALOADER.SENTINEL1_BANDS)
        self.sar_stream = UNet(cfg, n_channels=sar_in, n_classes=out, topology=topology, enable_outc=False)
        self.sar_in = sar_in
        self.sar_out_conv = OutConv(out_dim, out)

        # sentinel-2 optical unet stream
        optical_in = len(cfg.DATALOADER.SENTINEL2_BANDS)
        self.optical_stream = UNet(cfg, n_channels=optical_in, n_classes=out, topology=topology, enable_outc=False)
        self.optical_in = optical_in
        self.optical_out_conv = OutConv(out_dim, out)

        # out block combining unet outputs
        fusion_out_dim = 2 * out_dim
        self.fusion_out_conv = OutConv(fusion_out_dim, out)

    def forward(self, x_fusion):

        # sar
        x_sar = x_fusion[:, :self.sar_in, ]
        features_sar = self.sar_stream(x_sar)

        # optical
        x_optical = x_fusion[:, self.sar_in:, ]
        features_optical = self.optical_stream(x_optical)

        features_fusion = torch.cat((features_sar, features_optical), dim=1)
        logits_fusion = self.fusion_out_conv(features_fusion)

        if self.training:
            logits_sar = self.sar_out_conv(features_sar)
            logits_optical = self.optical_out_conv(features_optical)

            return logits_sar, logits_optical, logits_fusion

        else:
            return logits_fusion

    def fusion_features(self, x_fusion):

        # sar
        x_sar = x_fusion[:, :self.sar_in, ]
        features_sar = self.sar_stream(x_sar)

        # optical
        x_optical = x_fusion[:, self.sar_in:, ]
        features_optical = self.optical_stream(x_optical)

        features_fusion = torch.cat((features_sar, features_optical), dim=1)
        return features_fusion


# sub-parts of the U-Net model
class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(InConv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.detach().size()[2] - x1.detach().size()[2]
        diffX = x2.detach().size()[3] - x1.detach().size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
