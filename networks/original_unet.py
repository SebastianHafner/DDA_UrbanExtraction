import torch
import torch.nn as nn

# https://github.com/spctr01/UNet

# double 3x3 convolution
def dual_conv(in_channel, out_channel):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
    return conv


# crop the image(tensor) to equal size
# as shown in architecture image , half left side image is concated with right side image
def crop_tensor(target_tensor, tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2

    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]


class OriginalUNet(nn.Module):
    def __init__(self, cfg):
        super(OriginalUNet, self).__init__()

        self.cfg = cfg
        in_channels = cfg.MODEL.IN_CHANNELS
        out_channels = cfg.MODEL.OUT_CHANNELS

        # Left side (contracting path)
        self.dwn_conv1 = dual_conv(in_channels, 64)
        self.dwn_conv2 = dual_conv(64, 128)
        self.dwn_conv3 = dual_conv(128, 256)
        self.dwn_conv4 = dual_conv(256, 512)
        self.dwn_conv5 = dual_conv(512, 1024)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Right side  (expnsion path)
        # transpose convolution is used showna as green arrow in architecture image
        self.trans1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = dual_conv(1024, 512)
        self.trans2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = dual_conv(512, 256)
        self.trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = dual_conv(256, 128)
        self.trans4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = dual_conv(128, 64)

        # output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, image):
        # forward pass for Left side
        x1 = self.dwn_conv1(image)
        x2 = self.maxpool(x1)
        x3 = self.dwn_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.dwn_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.dwn_conv4(x6)
        x8 = self.maxpool(x7)
        x9 = self.dwn_conv5(x8)

        # forward pass for Right side
        x = self.trans1(x9)
        x = self.up_conv1(torch.cat([x, x7], 1))

        x = self.trans2(x)
        x = self.up_conv2(torch.cat([x, x5], 1))

        x = self.trans3(x)
        x = self.up_conv3(torch.cat([x, x3], 1))

        x = self.trans4(x)
        x = self.up_conv4(torch.cat([x, x1], 1))

        x = self.out(x)

        return x


if __name__ == '__main__':
    # image = torch.rand((1, 1, 572, 572))
    # model = Unet()
    # model(image)
    pass
