import segmentation_models_pytorch as smp


def ResNet(cfg):

    net = smp.Unet(
        cfg.MODEL.BACKBONE.TYPE,
        encoder_weights=cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS,
        in_channels=cfg.MODEL.IN_CHANNELS,
        classes=cfg.MODEL.OUT_CHANNELS,
        aux_params={'activation': None},
    )

    return net
