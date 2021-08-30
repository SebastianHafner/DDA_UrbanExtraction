import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
from pathlib import Path
from utils.augmentations import *
from utils.geotiff import *


# see https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(keras.utils.Sequence):

    def __init__(self, cfg, dataset: str, dim: tuple = (256, 256)):

        # Initialization
        self.cfg = cfg
        self.dataset = dataset
        self.root_dir = Path(cfg.DATASETS.PATH)
        self.batch_size = cfg.TRAINER.BATCH_SIZE
        self.shuffle = cfg.DATALOADER.SHUFFLE
        self.dim = dim
        self.dataset = dataset
        self.cities = cfg.DATASETS.TRAIN if dataset == 'train' else cfg.DATASETS.TEST

        # Loading samples for all cities in dataset
        self.samples = []
        for city in self.cities:
            samples_file = self.root_dir / city / 'samples.json'
            with open(str(samples_file)) as f:
                metadata = json.load(f)
            self.samples += metadata['samples']
        self.length = len(self.samples)

        # Get indices of selected bands for Sentinel-1 and Sentinel-2
        s1_bands = ['VV', 'VH']
        self.s1_indices = self._get_indices(s1_bands, cfg.DATALOADER.SENTINEL1_BANDS)
        s2_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        self.s2_indices = self._get_indices(s2_bands, cfg.DATALOADER.SENTINEL2_BANDS)
        self.n_channels = len(self.s1_indices) + len(self.s2_indices)

        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(batch_indices)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_indices: list):
        # Generates data containing batch_size samples  X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1), dtype=int)

        # Generate data
        for i, sample_index in enumerate(batch_indices):
            img, label = self.get_sample(sample_index)

            # Store img and corresponding label
            X[i, ], y[i, ] = img, label

        return X, y

    def get_sample(self, i: int):

        sample = self.samples[i]
        city, patch_id = sample['city'], sample['patch_id']

        # loading images
        if not any(self.cfg.DATALOADER.SENTINEL1_BANDS):  # only sentinel 2 features
            img = self._get_sentinel2_data(city, patch_id)
        elif not any(self.cfg.DATALOADER.SENTINEL2_BANDS):  # only sentinel 1 features
            img = self._get_sentinel1_data(city, patch_id)
        else:  # sentinel 1 and sentinel 2 features
            s1_img = self._get_sentinel1_data(city, patch_id)
            s2_img = self._get_sentinel2_data(city, patch_id)
            img = np.concatenate([s1_img, s2_img], axis=-1)

        label = self._get_label_data(city, patch_id)
        if self.dataset == 'train':
            img, label = self._transform(img, label)

        return img, label

    def _get_sentinel1_data(self, city, patch_id):
        file = self.root_dir / city / 'sentinel1' / f'sentinel1_{city}_{patch_id}.tif'
        img, _, _ = read_tif(file)
        img = img[:, :, self.s1_indices]
        img = np.nan_to_num(img).astype(np.float32)
        return img

    def _get_sentinel2_data(self, city, patch_id):
        file = self.root_dir / city / 'sentinel2' / f'sentinel2_{city}_{patch_id}.tif'
        img, _, _ = read_tif(file)
        img = img[:, :, self.s2_indices]
        img = np.nan_to_num(img).astype(np.float32)
        return img

    def _get_label_data(self, city, patch_id):
        label = self.cfg.DATALOADER.LABEL
        threshold = self.cfg.DATALOADER.LABEL_THRESH
        label_file = self.root_dir / city / label / f'{label}_{city}_{patch_id}.tif'
        img, transform, crs = read_tif(label_file)
        if threshold >= 0:
            img = img > threshold
        img = np.nan_to_num(img).astype(np.int)
        return img

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.cities)} cities.'

    def _transform(self, img, label):

        # Flips
        if self.cfg.AUGMENTATION.RANDOM_FLIP:
            flip_horizontally = np.random.choice([True, False])
            if flip_horizontally:
                img = np.flip(img, axis=1).copy()
                label = np.flip(label, axis=1).copy()
            flip_vertically = np.random.choice([True, False])
            if flip_vertically:
                img = np.flip(img, axis=0).copy()
                label = np.flip(label, axis=0).copy()

        # Rotations
        if self.cfg.AUGMENTATION.RANDOM_ROTATE:
            k = np.random.randint(1, 4)  # number of 90 degree rotations
            img = np.rot90(img, k, axes=(0, 1)).copy()
            label = np.rot90(label, k, axes=(0, 1)).copy()

        return img, label

    @ staticmethod
    def _get_indices(bands, selection):
        return [bands.index(band) for band in selection]


