from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn


class DenseFusionNet(nn.Module):

    def __init__(self, cfg):
        super(DenseFusionNet, self).__init__()
        self._cfg = cfg

        self.in_optical = len(cfg.DATALOADER.SENTINEL2_BANDS)
        self.in_sar = len(cfg.DATALOADER.SENTINEL1_BANDS)
        self.out = cfg.MODEL.OUT_CHANNELS

        # optical encoder stream
        self.inc_optical = InConv(self.in_optical, 64, DoubleConv)
        self.enc_block1_optical = Down(64, 128, DoubleConv)
        self.enc_block2_optical = Down(128, 256, DoubleConv)
        self.enc_block3_optical = Down(256, 512, DoubleConv)
        self.enc_block4_optical = Down(512, 512, DoubleConv)

        # sar stream
        self.inc_sar = InConv(self.in_sar, 64, DoubleConv)
        self.enc_block1_sar = Down(64, 128, DoubleConv)
        self.enc_block2_sar = Down(128, 256, DoubleConv)
        self.enc_block3_sar = Down(256, 512, DoubleConv)
        self.enc_block4_sar = Down(512, 512, DoubleConv)

        self.bottleneck_fusion = DoubleConv(1024, 512)

        # fusion decoder
        self.dec_block1 = DenseFusionUp(512, 512, 512, 256, DoubleConv)
        self.dec_block2 = DenseFusionUp(256, 256, 256, 128, DoubleConv)
        self.dec_block3 = DenseFusionUp(128, 128, 128, 64, DoubleConv)
        self.dec_block4 = DenseFusionUp(64, 64, 64, 64, DoubleConv)
        self.outc = OutConv(64, 1)

    def forward(self, x_in):

        x_in_sar = x_in[:, :self.in_sar, :, :]
        x_in_optical = x_in[:, self.in_sar:, :, :]

        # 64 filters
        x1_optical = self.inc_optical(x_in_optical)
        # 128 filters
        x2_optical = self.enc_block1_optical(x1_optical)
        # 256 filters
        x3_optical = self.enc_block2_optical(x2_optical)
        # 512 filters
        x4_optical = self.enc_block3_optical(x3_optical)
        # 512 filters
        x5_optical = self.enc_block4_optical(x4_optical)

        # 64 filters
        x1_sar = self.inc_sar(x_in_sar)
        # 128 filters
        x2_sar = self.enc_block1_optical(x1_sar)
        # 256 filters
        x3_sar = self.enc_block2_optical(x2_sar)
        # 512 filters
        x4_sar = self.enc_block3_optical(x3_sar)
        # 512 filters
        x5_sar = self.enc_block4_optical(x4_sar)

        # 1024 filters
        x5_concat = torch.cat([x5_optical, x5_sar], dim=1)
        # 512 filters
        x5_fused = self.bottleneck_fusion(x5_concat)
        #
        x6_fused = self.dec_block1(x5_fused, x4_optical, x4_sar)
        #
        x7_fused = self.dec_block2(x6_fused, x3_optical, x3_sar)
        #
        x8_fused = self.dec_block3(x7_fused, x2_optical, x2_sar)
        #
        x9_fused = self.dec_block4(x8_fused, x1_optical, x1_sar)

        x_out = self.outc(x9_fused)

        return x_out


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


class DenseFusionUp(nn.Module):
    def __init__(self, in_ch_fused, in_ch_optical, in_ch_sar, out_ch, conv_block):
        super(DenseFusionUp, self).__init__()

        self.in_ch_fused = in_ch_fused
        self.in_ch_optical = in_ch_optical
        self.in_ch_sar = in_ch_sar
        self.out_ch = out_ch
        self.up = nn.ConvTranspose2d(in_ch_fused, in_ch_fused, kernel_size=2, stride=2)
        self.conv = conv_block(in_ch_fused + in_ch_optical + in_ch_sar, out_ch)

    def forward(self, x1, x2_optical, x2_sar):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2_optical.detach().size()[2] - x1.detach().size()[2]
        diffX = x2_optical.detach().size()[3] - x1.detach().size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2_optical, x2_sar, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
