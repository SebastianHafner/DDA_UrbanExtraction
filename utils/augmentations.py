import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import cv2


def compose_transformations(cfg):
    transformations = []

    if cfg.AUGMENTATION.RANDOM_FLIP:
        transformations.append(RandomFlip())

    if cfg.AUGMENTATION.RANDOM_ROTATE:
        transformations.append(RandomRotate())

    if cfg.AUGMENTATION.COLOR_SHIFT:
        transformations.append(ColorShift())

    if cfg.AUGMENTATION.GAMMA_CORRECTION:
        transformations.append(GammaCorrection())

    if cfg.AUGMENTATION.DOWNSCALE:
        for f in cfg.AUGMENTATION.DOWNSCALE:
            transformations.append(DownScale(f))

    if cfg.AUGMENTATION.SENSOR_DROPOUT:
        transformations.append(SensorDropout(cfg))

    if cfg.AUGMENTATION.CHANNEL_DROPOUT:
        transformations.append(ChannelDropout(cfg))

    transformations.append(Numpy2Torch())

    return transforms.Compose(transformations)


class Numpy2Torch(object):
    def __call__(self, args):
        img, label = args
        img_tensor = TF.to_tensor(img)
        label_tensor = TF.to_tensor(label)
        return img_tensor, label_tensor


class RandomFlip(object):
    def __call__(self, args):
        img, label = args
        horizontal_flip = np.random.choice([True, False])
        vertical_flip = np.random.choice([True, False])

        if horizontal_flip:
            img = np.flip(img, axis=1)
            label = np.flip(label, axis=1)

        if vertical_flip:
            img = np.flip(img, axis=0)
            label = np.flip(label, axis=0)

        img = img.copy()
        label = label.copy()

        return img, label


class RandomRotate(object):
    def __call__(self, args):
        img, label = args
        k = np.random.randint(1, 4) # number of 90 degree rotations
        img = np.rot90(img, k, axes=(0, 1)).copy()
        label = np.rot90(label, k, axes=(0, 1)).copy()
        return img, label


class ColorShift(object):
    def __init__(self, min_factor: float = 0.5, max_factor: float = 1.5):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, args):
        img, label = args
        factors = np.random.uniform(self.min_factor, self.max_factor, img.shape[-1])
        img_rescaled = np.clip(img * factors[np.newaxis, np.newaxis, :], 0, 1).astype(np.float32)
        return img_rescaled, label


class GammaCorrection(object):
    def __init__(self, gain: float = 1, min_gamma: float = 0.25, max_gamma: float = 2):
        self.gain = gain
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def __call__(self, args):
        img, label = args
        gamma = np.random.uniform(self.min_gamma, self.max_gamma, img.shape[-1])
        img_gamma_corrected = np.clip(np.power(img,gamma[np.newaxis, np.newaxis, :]), 0, 1).astype(np.float32)
        return img_gamma_corrected, label


class DownScale(object):
    def __init__(self, factor: int):
        self.factor = factor

    def __call__(self, args):
        img, label = args
        m, n, _ = img.shape
        m_to, n_to = m // self.factor, n // self.factor
        img = cv2.resize(img, dsize=(m_to, n_to), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, dsize=(m, n), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, dsize=(m_to, n_to), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, dsize=(m, n), interpolation=cv2.INTER_NEAREST)
        return img, label


class SensorDropout(object):
    def __init__(self, cfg):
        assert(cfg.DATALOADER.MODE == 'fusion')
        self.split_index = len(cfg.DATALOADER.SENTINEL1_BANDS)

    def __call__(self, args):
        img, label = args
        s1_img = img[:, :, :self.split_index]
        s2_img = img[:, :, self.split_index:]
        dropout_layer = np.random.randint(0, 3)
        if dropout_layer == 1:
            s1_img[...] = 0
        if dropout_layer == 2:
            s2_img[...] = 0
        img = np.concatenate([s1_img, s2_img], axis=-1)
        return img, label


class ChannelDropout(object):
    def __init__(self, cfg):
        n_s1 = len(cfg.DATALOADER.SENTINEL1_BANDS)
        n_s2 = len(cfg.DATALOADER.SENTINEL2_BANDS)
        self.n = n_s1 + n_s2

    def __call__(self, args):
        img, label = args
        dropout_layer = np.random.randint(0, self.n + 1)
        # no dropout possible
        if not dropout_layer == self.n:
            img[:, :, dropout_layer] = 0
        return img, label


class ImageCrop(object):
    def __init__(self, crop_size: int):
        self.crop_size = crop_size

    def __call__(self, args):
        img, label = args
        m, n, _ = img.shape
        i, j = np.random.randint(0, m - self.crop_size), np.random.randint(0, n - self.crop_size)
        img_crop = img[i:i + self.crop_size, j:j + self.crop_size, ]
        label_crop = label[i:i + self.crop_size, j:j + self.crop_size, ]
        return img_crop, label_crop
