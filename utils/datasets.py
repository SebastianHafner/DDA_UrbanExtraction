import torch
from torchvision import transforms
from pathlib import Path
from abc import abstractmethod
import affine
import numpy as np
from utils import augmentations, geofiles, paths


class AbstractUrbanExtractionDataset(torch.utils.data.Dataset):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        dirs = paths.load_paths()
        self.dirs = dirs
        self.root_path = Path(dirs.DATASET)

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        s1_bands = ['VV', 'VH']
        self.s1_indices = self._get_indices(s1_bands, cfg.DATALOADER.SENTINEL1_BANDS)
        s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        self.s2_indices = self._get_indices(s2_bands, cfg.DATALOADER.SENTINEL2_BANDS)

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def _get_sentinel1_data(self, site, patch_id):
        file = self.root_path / site / 'sentinel1' / f'sentinel1_{site}_{patch_id}.tif'
        img, transform, crs = geofiles.read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_sentinel2_data(self, site, patch_id):
        file = self.root_path / site / 'sentinel2' / f'sentinel2_{site}_{patch_id}.tif'
        img, transform, crs = geofiles.read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_label_data(self, site, patch_id):
        label = self.cfg.DATALOADER.LABEL
        label_file = self.root_path / site / label / f'{label}_{site}_{patch_id}.tif'
        img, transform, crs = geofiles.read_tif(label_file)
        img = img > 0
        return np.nan_to_num(img).astype(np.float32), transform, crs

    @staticmethod
    def _get_indices(bands, selection):
        return [bands.index(band) for band in selection]


# dataset for urban extraction with building footprints
class UrbanExtractionDataset(AbstractUrbanExtractionDataset):

    def __init__(self, cfg, dataset: str, include_projection: bool = False, no_augmentations: bool = False,
                 include_unlabeled: bool = True):
        super().__init__(cfg)

        self.dataset = dataset
        if dataset == 'training':
            self.sites = list(cfg.DATASETS.TRAINING)
            # using parameter include_unlabeled to overwrite config
            if include_unlabeled and cfg.DATALOADER.INCLUDE_UNLABELED:
                self.sites += cfg.DATASETS.UNLABELED
        elif dataset == 'validation':
            self.sites = list(cfg.DATASETS.VALIDATION)
        else:  # used to load only 1 city passed as dataset
            self.sites = [dataset]

        self.no_augmentations = no_augmentations
        if no_augmentations:
            self.transform = transforms.Compose([augmentations.Numpy2Torch()])
        else:
            self.transform = augmentations.compose_transformations(cfg)

        self.samples = []
        for site in self.sites:
            samples_file = self.root_path / site / 'samples.json'
            metadata = geofiles.load_json(samples_file)
            samples = metadata['samples']
            # making sure unlabeled data is not used as labeled when labels exist
            if include_unlabeled and site in cfg.DATASETS.UNLABELED:
                for sample in samples:
                    sample['is_labeled'] = False
            self.samples += samples
        self.length = len(self.samples)
        self.n_labeled = len([s for s in self.samples if s['is_labeled']])

        self.include_projection = include_projection

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]

        site = sample['site']
        patch_id = sample['patch_id']
        is_labeled = sample['is_labeled']

        # loading images
        mode = self.cfg.DATALOADER.MODE
        if mode == 'optical':
            img, geotransform, crs = self._get_sentinel2_data(site, patch_id)
        elif mode == 'sar':
            img, geotransform, crs = self._get_sentinel1_data(site, patch_id)
        else:  # fusion baby!!!
            s1_img, geotransform, crs = self._get_sentinel1_data(site, patch_id)
            s2_img, _, _ = self._get_sentinel2_data(site, patch_id)
            img = np.concatenate([s1_img, s2_img], axis=-1)

        # create dummy label if unlabeled
        if is_labeled:
            label, _, _ = self._get_label_data(site, patch_id)
        else:
            label = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)

        img, label = self.transform((img, label))

        item = {
            'x': img,
            'y': label,
            'site': site,
            'patch_id': patch_id,
            'is_labeled': sample['is_labeled'],
            'image_weight': np.float(sample['img_weight']),
        }

        if self.include_projection:
            item['transform'] = geotransform
            item['crs'] = crs

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        labeled_perc = self.n_labeled / self.length * 100
        return f'Dataset with {self.length} samples ({labeled_perc:.1f} % labeled) across {len(self.sites)} sites.'


# dataset for urban extraction with building footprints
class UrbanExtractionSpacenet7S1S2Dataset(AbstractUrbanExtractionDataset):

    def __init__(self, cfg, dataset: str, include_projection: bool = False, no_augmentations: bool = False,
                 include_unlabeled: bool = True):
        super().__init__(cfg)

        self.dataset = dataset
        if dataset == 'training':
            self.sites = list(cfg.DATASETS.TRAINING)
            # using parameter include_unlabeled to overwrite config
            if include_unlabeled and cfg.DATALOADER.INCLUDE_UNLABELED:
                self.sites += cfg.DATASETS.UNLABELED
        elif dataset == 'validation':
            self.sites = list(cfg.DATASETS.VALIDATION)
        else:  # used to load only 1 city passed as dataset
            self.sites = [dataset]

        self.no_augmentations = no_augmentations
        if no_augmentations:
            self.transform = transforms.Compose([augmentations.Numpy2Torch()])
        else:
            self.transform = augmentations.compose_transformations(cfg)

        self.samples = []
        for site in self.sites:
            samples_file = self.root_path / site / 'samples.json'
            metadata = geofiles.load_json(samples_file)
            samples = metadata['samples']
            # making sure unlabeled data is not used as labeled when labels exist
            if include_unlabeled and site in cfg.DATASETS.UNLABELED:
                for sample in samples:
                    sample['is_labeled'] = False
            self.samples += samples
        for sample in self.samples:
            sample['dataset'] = 'urban_dataset'

        # include samples from spacenet7 s1s2
        if include_unlabeled:
            metadata_file_spacenet7s1s2 = Path(self.dirs.DATASET_SPACENET7S1S2) / 'metadata.json'
            metadata_spacenet7s1s2 = geofiles.load_json(metadata_file_spacenet7s1s2)
            for aoi_id in metadata_spacenet7s1s2['aois'].keys():
                aoi_timestamps = metadata_spacenet7s1s2['aois'][aoi_id]
                for timestamp in aoi_timestamps:
                    year, month, _, s1, s2 = timestamp
                    if cfg.DATALOADER.MODE == 'optical' and s2:
                        add_to_samples = True
                    elif cfg.DATALOADER.MODE == 'sar' and s1:
                        add_to_samples = True
                    elif cfg.DATALOADER.MODE == 'fusion' and (s1 and s2):
                        add_to_samples = True
                    else:
                        add_to_samples = False
                    if add_to_samples:
                        sample = {
                            'dataset': 'spacenet7s1s2',
                            'aoi_id': aoi_id,
                            'year': year,
                            'month': month,
                            'is_labeled': False,
                        }
                        self.samples.append(sample)

        self.length = len(self.samples)
        self.n_labeled = len([s for s in self.samples if s['is_labeled']])

        self.include_projection = include_projection

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]
        is_labeled = sample['is_labeled']
        dataset = sample['dataset']
        patch_id = sample['patch_id'] if dataset == 'urban_dataset' else '-'
        site = sample['site'] if dataset == 'urban_dataset' else sample['aoi_id']
        img_weight = np.float(sample['img_weight']) if dataset == 'urban_dataset' else np.float(0)
        mode = self.cfg.DATALOADER.MODE

        if dataset == 'urban_dataset':
            if mode == 'optical':
                img, geotransform, crs = self._get_sentinel2_data(site, patch_id)
            elif mode == 'sar':
                img, geotransform, crs = self._get_sentinel1_data(site, patch_id)
            else:
                s1_img, geotransform, crs = self._get_sentinel1_data(site, patch_id)
                s2_img, _, _ = self._get_sentinel2_data(site, patch_id)
                img = np.concatenate([s1_img, s2_img], axis=-1)
            if is_labeled:
                label, _, _ = self._get_label_data(site, patch_id)
            else:
                label = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)

        else:
            year, month = sample['year'], sample['month']
            if mode == 'optical':
                img, geotransform, crs = self._get_spacenet7s1s2_data('sentinel2', site, year, month)
            elif mode == 'sar':
                img, geotransform, crs = self._get_spacenet7s1s2_data('sentinel1', site, year, month)
            else:
                s1_img, geotransform, crs = self._get_spacenet7s1s2_data('sentinel1', site, year, month)
                s2_img, _, _ = img, geotransform, crs = self._get_spacenet7s1s2_data('sentinel2', site, year, month)
                img = np.concatenate([s1_img, s2_img], axis=-1)

            # cropping spacenet7 image
            crop_size = 256
            height, width, _ = img.shape
            crop_limit_x = width - crop_size
            crop_limit_y = height - crop_size
            x = np.random.randint(0, crop_limit_x)
            y = np.random.randint(0, crop_limit_y)
            img = img[y:y + crop_size, x:x + crop_size, ]
            label = np.zeros((crop_size, crop_size, 1), dtype=np.float32)

        img, label = self.transform((img, label))
        item = {
            'x': img,
            'y': label,
            'site': site,
            'patch_id': patch_id,
            'is_labeled': sample['is_labeled'],
            'image_weight': img_weight,
        }

        if self.include_projection:
            item['transform'] = geotransform
            item['crs'] = crs

        return item

    def _get_spacenet7s1s2_data(self, sensor: str, aoi_id: str, year: int, month: int):
        fname = f'{sensor}_{aoi_id}_{year}_{month:02d}.tif'
        file = Path(self.dirs.DATASET_SPACENET7S1S2) / aoi_id / sensor / fname
        img, transform, crs = geofiles.read_tif(file)
        band_indices = self.s1_indices if sensor == 'sentinel1' else self.s2_indices
        img = img[:, :, band_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def __len__(self):
        return self.length

    def __str__(self):
        labeled_perc = self.n_labeled / self.length * 100
        return f'Dataset with {self.length} samples ({labeled_perc:.1f} % labeled) across {len(self.sites)} sites.'


class AbstractSpaceNet7Dataset(torch.utils.data.Dataset):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        dirs = paths.load_paths()
        self.root_path = Path(dirs.DATASET) / 'sn7'

        # getting patches
        samples_file = self.root_path / 'samples.json'
        metadata = geofiles.load_json(samples_file)
        self.samples = metadata['samples']
        self.group_names = metadata['group_names']
        self.length = len(self.samples)
        s1_bands = metadata['sentinel1_features']
        s2_bands = metadata['sentinel2_features']

        self.transform = transforms.Compose([augmentations.Numpy2Torch()])

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        self.s1_indices = self._get_indices(s1_bands, cfg.DATALOADER.SENTINEL1_BANDS)
        self.s2_indices = self._get_indices(s2_bands, cfg.DATALOADER.SENTINEL2_BANDS)

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    def _get_sentinel1_data(self, aoi_id):
        file = self.root_path / 'sentinel1' / f'sentinel1_{aoi_id}.tif'
        img, transform, crs = geofiles.read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_sentinel2_data(self, aoi_id):
        file = self.root_path / 'sentinel2' / f'sentinel2_{aoi_id}.tif'
        img, transform, crs = geofiles.read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_label_data(self, aoi_id):
        label = self.cfg.DATALOADER.LABEL
        label_file = self.root_path / label / f'{label}_{aoi_id}.tif'
        img, transform, crs = geofiles.read_tif(label_file)
        img = img > 0
        return np.nan_to_num(img).astype(np.float32), transform, crs


    def get_index(self, aoi_id: str):
        for i, sample in enumerate(self.samples):
            if sample['aoi_id'] == aoi_id:
                return i

    @staticmethod
    def _get_indices(bands, selection):
        return [bands.index(band) for band in selection]

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


# dataset for urban extraction with building footprints
class SpaceNet7Dataset(AbstractSpaceNet7Dataset):

    def __init__(self, cfg):
        super().__init__(cfg)

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]
        aoi_id = sample['aoi_id']
        group = sample['group']
        group_name = self.group_names[str(group)]

        # loading images
        mode = self.cfg.DATALOADER.MODE
        if mode == 'optical':
            img, _, _ = self._get_sentinel2_data(aoi_id)
        elif mode == 'sar':
            img, _, _ = self._get_sentinel1_data(aoi_id)
        else:  # fusion baby!!!
            s1_img, _, _ = self._get_sentinel1_data(aoi_id)
            s2_img, _, _ = self._get_sentinel2_data(aoi_id)

            img = np.concatenate([s1_img, s2_img], axis=-1)

        label, geotransform, crs = self._get_label_data(aoi_id)
        img, label = self.transform((img, label))

        item = {
            'x': img,
            'y': label,
            'aoi_id': aoi_id,
            'country': sample['country'],
            'group': group,
            'group_name': group_name,
            'year': sample['year'],
            'month': sample['month'],
            'transform': geotransform,
            'crs': crs
        }

        return item


# dataset for classifying a scene
class TilesInferenceDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, site: str):
        super().__init__()

        self.cfg = cfg
        self.site = site

        dirs = paths.load_paths()
        self.root_dir = Path(dirs.DATASET)
        self.transform = transforms.Compose([augmentations.Numpy2Torch()])

        # getting all files
        samples_file = self.root_dir / site / 'samples.json'
        metadata = geofiles.load_json(samples_file)
        self.samples = metadata['samples']
        self.length = len(self.samples)

        self.patch_size = metadata['patch_size']

        # computing extent
        patch_ids = [s['patch_id'] for s in self.samples]
        self.coords = [[int(c) for c in patch_id.split('-')] for patch_id in patch_ids]
        self.max_y = max([c[0] for c in self.coords])
        self.max_x = max([c[1] for c in self.coords])

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        self.s1_indices = self._get_indices(metadata['sentinel1_features'], cfg.DATALOADER.SENTINEL1_BANDS)
        self.s2_indices = self._get_indices(metadata['sentinel2_features'], cfg.DATALOADER.SENTINEL2_BANDS)
        if cfg.DATALOADER.MODE == 'sar':
            self.n_features = len(self.s1_indices)
        elif cfg.DATALOADER.MODE == 'optical':
            self.n_features = len(self.s2_indices)
        else:
            self.n_features = len(self.s1_indices) + len(self.s2_indices)

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]
        patch_id_center = sample['patch_id']

        y_center, x_center = patch_id_center.split('-')
        y_center, x_center = int(y_center), int(x_center)

        extended_patch = np.zeros((3 * self.patch_size, 3 * self.patch_size, self.n_features), dtype=np.float32)

        for i in range(3):
            for j in range(3):
                y = y_center + (i - 1) * self.patch_size
                x = x_center + (j - 1) * self.patch_size
                patch_id = f'{y:010d}-{x:010d}'
                if self._is_valid_patch_id(patch_id):
                    patch = self._load_patch(patch_id)
                else:
                    patch = np.zeros((self.patch_size, self.patch_size, self.n_features), dtype=np.float32)
                i_start = i * self.patch_size
                i_end = (i + 1) * self.patch_size
                j_start = j * self.patch_size
                j_end = (j + 1) * self.patch_size
                extended_patch[i_start:i_end, j_start:j_end, :] = patch

        if sample['is_labeled']:
            label, _, _ = self._get_label_data(patch_id_center)
        else:
            # dummy_label = np.zeros((extended_patch.shape[0], extended_patch.shape[1], 1), dtype=np.float32)
            dummy_label = np.zeros((self.patch_size, self.patch_size, 1), dtype=np.float32)
            label = dummy_label
        extended_patch, label = self.transform((extended_patch, label))

        item = {
            'x': extended_patch,
            'y': label,
            'i': y_center,
            'j': x_center,
            'site': self.site,
            'patch_id': patch_id_center,
            'is_labeled': sample['is_labeled']
        }

        return item

    def _is_valid_patch_id(self, patch_id):
        patch_ids = [s['patch_id'] for s in self.samples]
        return True if patch_id in patch_ids else False

    def _load_patch(self, patch_id):
        mode = self.cfg.DATALOADER.MODE
        if mode == 'optical':
            img, _, _ = self._get_sentinel2_data(patch_id)
        elif mode == 'sar':
            img, _, _ = self._get_sentinel1_data(patch_id)
        else:  # fusion baby!!!
            s1_img, _, _ = self._get_sentinel1_data(patch_id)
            s2_img, _, _ = self._get_sentinel2_data(patch_id)
            img = np.concatenate([s1_img, s2_img], axis=-1)
        return img

    def _get_sentinel1_data(self, patch_id):
        file = self.root_dir / self.site / 'sentinel1' / f'sentinel1_{self.site}_{patch_id}.tif'
        img, transform, crs = geofiles.read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_sentinel2_data(self, patch_id):
        file = self.root_dir / self.site / 'sentinel2' / f'sentinel2_{self.site}_{patch_id}.tif'
        img, transform, crs = geofiles.read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_label_data(self, patch_id):
        label = self.cfg.DATALOADER.LABEL
        threshold = self.cfg.DATALOADER.LABEL_THRESH

        label_file = self.root_dir / self.site / label / f'{label}_{self.site}_{patch_id}.tif'
        img, transform, crs = geofiles.read_tif(label_file)
        if threshold >= 0:
            img = img > threshold

        return np.nan_to_num(img).astype(np.float32), transform, crs

    def get_arr(self, dtype=np.uint8):
        height = self.max_y + self.patch_size
        width = self.max_x + self.patch_size
        return np.zeros((height, width, 1), dtype=dtype)

    def get_geo(self):
        patch_id = f'{0:010d}-{0:010d}'
        # in training and validation set patches with no BUA were not downloaded -> top left patch may not be available
        if self._is_valid_patch_id(patch_id):
            _, transform, crs = self._get_sentinel1_data(patch_id)
        else:
            # use first patch and covert transform to that of uupper left patch
            patch = self.samples[0]
            patch_id = patch['patch_id']
            i, j = patch_id.split('-')
            i, j = int(i), int(j)
            _, transform, crs = self._get_sentinel1_data(patch_id)
            x_spacing, x_whatever, x_start, y_whatever, y_spacing, y_start, *_ = transform
            x_start -= (x_spacing * j)
            y_start -= (y_spacing * i)
            transform = affine.Affine(x_spacing, x_whatever, x_start, y_whatever, y_spacing, y_start)
        return transform, crs

    @staticmethod
    def _get_indices(bands, selection):
        return [bands.index(band) for band in selection]

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.sites)} sites.'


# dataset for continuous urban change detection project
class InferenceCUCDDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, include_geo: bool = False):
        super().__init__()

        self.cfg = cfg
        self.include_geo = include_geo

        dirs = paths.load_paths()
        self.root_path = Path(dirs.DATASET_SPACENET7S1S2)
        self.metadata = geofiles.load_json(self.root_path / 'metadata.json')

        self.mode = cfg.DATALOADER.MODE

        self.samples = []
        for aoi_id in self.metadata['aois'].keys():
            for sample in self.metadata['aois'][aoi_id]:
                year, month, mask, s1, s2 = sample
                sample = {
                    'aoi_id': aoi_id,
                    'year': year,
                    'month': month,
                    'has_mask': mask,
                }
                if self.mode == 'sar' and s1:
                    self.samples.append(sample)
                if self.mode == 'optical' and s2:
                    self.samples.append(sample)
                if self.mode == 'fusion' and s1 and s2:
                    self.samples.append(sample)
        self.length = len(self.samples)

        self.transform = transforms.Compose([augmentations.Numpy2Torch()])

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        s1_bands = self.metadata['s1_bands']
        s2_bands = self.metadata['s2_bands']
        # TODO: fix this without breaking the rest of the code :)
        s2_bands[5] = 'B7'
        self.s1_indices = self._get_indices(s1_bands, cfg.DATALOADER.SENTINEL1_BANDS)
        self.s2_indices = self._get_indices(s2_bands, cfg.DATALOADER.SENTINEL2_BANDS)

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]
        aoi_id, year, month, has_mask = sample['aoi_id'], sample['year'], sample['month'], sample['has_mask']

        if self.mode == 'sar':
            img, _, _ = self._get_cucd_sentinel1_data(aoi_id, year, month)
        elif self.mode == 'optical':
            img, _, _ = self._get_cucd_sentinel2_data(aoi_id, year, month)
        else:
            s1_img, _, _ = self._get_cucd_sentinel1_data(aoi_id, year, month)
            s2_img, _, _ = self._get_cucd_sentinel2_data(aoi_id, year, month)
            img = np.concatenate([s1_img, s2_img], axis=-1)

        label, geotransform, crs = self._get_cucd_label_data(aoi_id, year, month)
        if has_mask:
            mask, _, _ = self._get_cucd_mask_data(aoi_id, year, month)
            label[mask] = np.NaN

        img, label = self.transform((img, label))
        item = {
            'x': img,
            'y': label,
            'aoi_id': aoi_id,
            'year': year,
            'month': month,
            'has_mask': has_mask
        }

        if self.include_geo:
            item['transform'] = geotransform
            item['crs'] = crs

        return item

    def _get_cucd_sentinel1_data(self, aoi_id: str, year: int, month: int):
        file = self.root_path / aoi_id / 'sentinel1' / f'sentinel1_{aoi_id}_{year}_{month:02}.tif'
        img, transform, crs = geofiles.read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_cucd_sentinel2_data(self, aoi_id: str, year: int, month: int):
        file = self.root_path / aoi_id / 'sentinel2' / f'sentinel2_{aoi_id}_{year}_{month:02}.tif'
        img, transform, crs = geofiles.read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_cucd_label_data(self, aoi_id: str, year: int, month: int):
        file = self.root_path / aoi_id / 'buildings' / f'buildings_{aoi_id}_{year}_{month:02}.tif'
        img, transform, crs = geofiles.read_tif(file)
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_cucd_mask_data(self, aoi_id: str, year: int, month: int):
        file = self.root_path / aoi_id / f'masks_{aoi_id}.tif'
        img, transform, crs = geofiles.read_tif(file)
        index = self._get_cucd_mask_index(aoi_id, year, month)
        return img[:, :, index], transform, crs

    def _get_cucd_mask_index(self, aoi_id: str, year: int, month: int):
        md = self.metadata['aois'][aoi_id]
        md_masked = [[y, m, mask, *_] for y, m, mask, *_ in md if mask]
        for i, (y, m, *_) in enumerate(md_masked):
            if year == y and month == m:
                return i

    @staticmethod
    def _get_indices(bands, selection):
        return [bands.index(band) for band in selection]

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'
