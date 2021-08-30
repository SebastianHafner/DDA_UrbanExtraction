import torch
from torchvision import transforms

import json

from pathlib import Path
from abc import abstractmethod
from utils.augmentations import *
from utils.geotiff import *
import affine


class AbstractUrbanExtractionDataset(torch.utils.data.Dataset):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.root_path = Path(cfg.DATASETS.PATH)

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        s1_bands = ['VV', 'VH']
        self.s1_indices = self._get_indices(s1_bands, cfg.DATALOADER.SENTINEL1_BANDS)
        s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B6', 'B8', 'B8A', 'B11', 'B12']
        self.s2_indices = self._get_indices(s2_bands, cfg.DATALOADER.SENTINEL2_BANDS)

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def _get_sentinel1_data(self, site, patch_id):
        file = self.root_path / site / 'sentinel1' / f'sentinel1_{site}_{patch_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_sentinel2_data(self, site, patch_id):
        file = self.root_path / site / 'sentinel2' / f'sentinel2_{site}_{patch_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_auxiliary_data(self, aux_input, site, patch_id):
        file = self.root_path / site / aux_input / f'{aux_input}_{site}_{patch_id}.tif'
        img, transform, crs = read_tif(file)
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_label_data(self, site, patch_id):
        label = self.cfg.DATALOADER.LABEL
        threshold = self.cfg.DATALOADER.LABEL_THRESH

        label_file = self.root_path / site / label / f'{label}_{site}_{patch_id}.tif'
        img, transform, crs = read_tif(label_file)
        if threshold >= 0:
            img = img > threshold

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
        self.transform = transforms.Compose([Numpy2Torch()]) if no_augmentations else compose_transformations(cfg)

        self.samples = []
        for site in self.sites:
            samples_file = self.root_path / site / 'samples.json'
            metadata = load_json(samples_file)
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

        # here we load all auxiliary inputs and append them to features
        aux_inputs = self.cfg.DATALOADER.AUXILIARY_INPUTS
        for aux_input in aux_inputs:
            aux_img, _, _ = self._get_auxiliary_data(aux_input, site, patch_id)
            img = np.concatenate([aux_img, img], axis=-1)

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


# dataset for selfsupervised pretraining with urban extraction dataset
class SelfsupervisedUrbanExtractionDataset(AbstractUrbanExtractionDataset):

    def __init__(self, cfg, dataset: str, no_augmentations: bool = False):
        super().__init__(cfg)

        self.sites = list(cfg.DATASETS.PRETRAINING) if dataset == 'training' else list(cfg.DATASETS.VALIDATION)
        self.no_augmentations = no_augmentations
        self.transform = transforms.Compose([Numpy2Torch()]) if no_augmentations else compose_transformations(cfg)

        self.samples = []
        for site in self.sites:
            samples_file = self.root_path / site / 'samples.json'
            metadata = load_json(samples_file)
            samples = metadata['samples']
            self.samples += samples
        self.length = len(self.samples)

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]

        site = sample['site']
        patch_id = sample['patch_id']

        s2_img, _, _ = self._get_sentinel2_data(site, patch_id)
        s1_img, _, _ = self._get_sentinel1_data(site, patch_id)

        s2_img, s1_img = self.transform((s2_img, s1_img))

        item = {
            'x': s2_img,
            'y': s1_img,
            'site': site,
            'patch_id': patch_id,
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.sites)} sites.'


class AbstractSpaceNet7Dataset(torch.utils.data.Dataset):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.root_path = Path(cfg.DATASETS.TESTING)

        # getting patches
        samples_file = self.root_path / 'samples.json'
        metadata = load_json(samples_file)
        self.samples = metadata['samples']
        self.group_names = metadata['group_names']
        self.length = len(self.samples)
        s1_bands = metadata['sentinel1_features']
        s2_bands = metadata['sentinel2_features']

        self.transform = transforms.Compose([Numpy2Torch()])

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        self.s1_indices = self._get_indices(s1_bands, cfg.DATALOADER.SENTINEL1_BANDS)
        self.s2_indices = self._get_indices(s2_bands, cfg.DATALOADER.SENTINEL2_BANDS)

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    def _get_sentinel1_data(self, aoi_id):
        file = self.root_path / 'sentinel1' / f'sentinel1_{aoi_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_sentinel2_data(self, aoi_id):
        file = self.root_path / 'sentinel2' / f'sentinel2_{aoi_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_label_data(self, aoi_id):

        label = self.cfg.DATALOADER.LABEL
        threshold = self.cfg.DATALOADER.LABEL_THRESH

        label_file = self.root_path / label / f'{label}_{aoi_id}.tif'
        img, transform, crs = read_tif(label_file)
        if threshold >= 0:
            img = img > threshold

        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_auxiliary_data(self, aux_input, aoi_id):
        file = self.root_path / aux_input / f'{aux_input}_{aoi_id}.tif'
        img, transform, crs = read_tif(file)
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

        aux_inputs = self.cfg.DATALOADER.AUXILIARY_INPUTS
        for aux_input in aux_inputs:
            aux_img, _, _ = self._get_auxiliary_data(aux_input, aoi_id)
            img = np.concatenate([aux_img, img], axis=-1)

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


# dataset for continuous urban change detection project
class CUCDDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, dataset_name: Path = None, include_geo: bool = False):
        super().__init__()

        self.cfg = cfg
        self.dataset_name = dataset_name
        self.include_geo = include_geo

        if dataset_name is None:
            self.root_path = Path(cfg.DATASETS.TESTING)
        else:
            self.root_path = Path(f'/storage/shafner/continuous_urban_change_detection/{dataset_name}/')
        metadata_file = self.root_path / 'metadata.json'
        self.metadata = load_json(metadata_file)

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

        self.transform = transforms.Compose([Numpy2Torch()])

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        s1_bands = self.metadata['s1_bands']
        s2_bands = self.metadata['s2_bands']
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
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_cucd_sentinel2_data(self, aoi_id: str, year: int, month: int):
        file = self.root_path / aoi_id / 'sentinel2' / f'sentinel2_{aoi_id}_{year}_{month:02}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_cucd_label_data(self, aoi_id: str, year: int, month: int):
        file = self.root_path / aoi_id / 'buildings' / f'buildings_{aoi_id}_{year}_{month:02}.tif'
        img, transform, crs = read_tif(file)
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_cucd_mask_data(self, aoi_id: str, year: int, month: int):
        file = self.root_path / aoi_id / f'masks_{aoi_id}.tif'
        img, transform, crs = read_tif(file)
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


# dataset for continuous urban change detection project
class CUCDFinetuningDataset(AbstractSpaceNet7Dataset):

    def __init__(self, cfg, cucd_dataset: str, run_type: str):
        super().__init__(cfg)

        self.root_path = Path(cfg.DATASETS.PATH)

        self.cucd_dataset = cucd_dataset
        self.run_type = run_type
        self.cucd_dataset_path = Path('/storage/shafner/continuous_urban_change_detection/') / cucd_dataset
        metadata_file = self.cucd_dataset_path / 'metadata.json'
        self.metadata = load_json(metadata_file)

        self.samples = []
        for aoi_id in self.metadata['aois'].keys():
            for sample in self.metadata['aois'][aoi_id]:
                year, month, mask, s1, s2 = sample
                if s1 and s2:
                    sample = {
                        'dataset': cucd_dataset,
                        'aoi_id': aoi_id,
                        'year': year,
                        'month': month,
                        'has_mask': mask,
                        's1': s1,
                        's2': s2
                    }
                    self.samples.append(sample)

        if run_type == 'training':
            training_sites = list(cfg.DATASETS.TRAINING)
            for site in training_sites:
                samples_file = self.root_path / site / 'samples.json'
                metadata = load_json(samples_file)
                samples = metadata['samples']
                for sample in samples:
                    sample['dataset'] = 'urban_extraction'
                    self.samples.append(sample)

        self.length = len(self.samples)

        if run_type == 'training':
            if cfg.FINETUNING.AUGMENTATIONS:
                self.transform = compose_transformations(cfg)
                self.transform.transforms.insert(0, ImageCrop(cfg.FINETUNING.CROP_SIZE))
            else:
                self.transform = transforms.Compose([ImageCrop(cfg.FINETUNING.CROP_SIZE), Numpy2Torch()])
        else:
            self.transform = transforms.Compose([Numpy2Torch()])

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]
        if sample['dataset'] == 'urban_extraction':
            site = sample['site']
            patch_id = sample['patch_id']

            s1_img, geotransform, crs = self._get_sentinel1_data(site, patch_id)
            s2_img, _, _ = self._get_sentinel2_data(site, patch_id)
            img = np.concatenate([s1_img, s2_img], axis=-1)

            label, _, _ = self._get_label_data(site, patch_id)
            is_labeled = True
        else:
            aoi_id, year, month, has_mask = sample['aoi_id'], sample['year'], sample['month'], sample['has_mask']

            s1_img, _, _ = self._get_cucd_sentinel1_data(aoi_id, year, month)
            s2_img, _, _ = self._get_cucd_sentinel2_data(aoi_id, year, month)
            img = np.concatenate([s1_img, s2_img], axis=-1)

            label, geotransform, crs = self._get_cucd_label_data(aoi_id, year, month)
            if has_mask:
                mask, _, _ = self._get_cucd_mask_data(aoi_id, year, month)
                label[mask] = np.NaN
            is_labeled = False

        img, label = self.transform((img, label))
        item = {
            'x': img,
            'y': label,
            'is_labeled': is_labeled
        }
        return item

    def _get_cucd_sentinel1_data(self, aoi_id: str, year: int, month: int):
        file = self.cucd_dataset_path / aoi_id / 'sentinel1' / f'sentinel1_{aoi_id}_{year}_{month:02}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_cucd_sentinel2_data(self, aoi_id: str, year: int, month: int):
        file = self.cucd_dataset_path / aoi_id / 'sentinel2' / f'sentinel2_{aoi_id}_{year}_{month:02}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_cucd_label_data(self, aoi_id: str, year: int, month: int):
        file = self.cucd_dataset_path / aoi_id / 'buildings' / f'buildings_{aoi_id}_{year}_{month:02}.tif'
        img, transform, crs = read_tif(file)
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_cucd_mask_data(self, aoi_id: str, year: int, month: int):
        file = self.cucd_dataset_path / aoi_id / f'masks_{aoi_id}.tif'
        img, transform, crs = read_tif(file)
        index = self._get_cucd_mask_index(aoi_id, year, month)
        return img[:, :, index], transform, crs

    def _get_cucd_mask_index(self, aoi_id: str, year: int, month: int):
        md = self.metadata['aois'][aoi_id]
        md_masked = [[y, m, mask, *_] for y, m, mask, *_ in md if mask]
        for i, (y, m, *_) in enumerate(md_masked):
            if year == y and month == m:
                return i

    def _get_sentinel1_data(self, site, patch_id):
        file = self.root_path / site / 'sentinel1' / f'sentinel1_{site}_{patch_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_sentinel2_data(self, site, patch_id):
        file = self.root_path / site / 'sentinel2' / f'sentinel2_{site}_{patch_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_label_data(self, site, patch_id):
        label = self.cfg.DATALOADER.LABEL
        threshold = self.cfg.DATALOADER.LABEL_THRESH

        label_file = self.root_path / site / label / f'{label}_{site}_{patch_id}.tif'
        img, transform, crs = read_tif(label_file)
        if threshold >= 0:
            img = img > threshold

        return np.nan_to_num(img).astype(np.float32), transform, crs





# dataset for selfsupervised pretraining with spacenet 7 dataset
class SelfsupervisedSpaceNet7Dataset(AbstractSpaceNet7Dataset):

    def __init__(self, cfg):
        super().__init__(cfg)

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]
        aoi_id = sample['aoi_id']
        group = sample['group']
        group_name = self.group_names[str(group)]

        s2_img, _, _ = self._get_sentinel2_data(aoi_id)
        s1_img, _, _ = self._get_sentinel1_data(aoi_id)

        s2_img, s1_img = self.transform((s2_img, s1_img))

        item = {
            'x': s2_img,
            'y': s1_img,
            'aoi_id': aoi_id,
            'country': sample['country'],
            'group': group,
            'group_name': group_name,
            'year': sample['year'],
            'month': sample['month'],
        }

        return item


# dataset for classifying a scene
class SceneInferenceDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, s1_file: Path = None, s2_file: Path = None, patch_size: int = 256,
                 s1_bands: list = None, s2_bands: list = None):
        super().__init__()

        self.cfg = cfg
        self.s1_file = s1_file
        self.s2_file = s2_file
        assert(s1_file.exists() and s2_file.exists())

        self.transform = transforms.Compose([Numpy2Torch()])

        ref_file = s1_file if s1_file is not None else s2_file
        arr, self.geotransform, self.crs = read_tif(ref_file)
        self.height, self.width, _ = arr.shape

        self.patch_size = patch_size
        self.rf = 128
        self.n_rows = (self.height - self.rf) // patch_size
        self.n_cols = (self.width - self.rf) // patch_size
        self.length = self.n_rows * self.n_cols

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        if s1_bands is None:
            s1_bands = ['VV_mean', 'VV_stdDev', 'VH_mean', 'VH_stdDev']
        selected_features_sentinel1 = cfg.DATALOADER.SENTINEL1_BANDS
        self.s1_feature_selection = self._get_feature_selection(s1_bands, selected_features_sentinel1)
        if s2_bands is None:
            s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        selected_features_sentinel2 = cfg.DATALOADER.SENTINEL2_BANDS
        self.s2_feature_selection = self._get_feature_selection(s2_bands, selected_features_sentinel2)

        # loading image
        mode = self.cfg.DATALOADER.MODE
        if mode == 'optical':
            img, _, _ = read_tif(self.s2_file)
            img = img[:, :, self.s2_feature_selection]
        elif mode == 'sar':
            img, _, _ = read_tif(self.s1_file)
            img = img[:, :, self.s1_feature_selection]
        else:  # fusion
            s1_img, _, _ = read_tif(self.s1_file)
            s1_img = s1_img[:, :, self.s1_feature_selection]
            s2_img, _, _ = read_tif(self.s2_file)
            s2_img = s2_img[:, :, self.s2_feature_selection]
            img = np.concatenate([s1_img, s2_img], axis=-1)
        self.img = img

    def __getitem__(self, index):

        i_start = index // self.n_cols * self.patch_size
        j_start = index % self.n_cols * self.patch_size
        # check for border cases and add padding accordingly
        # top left corner
        if i_start == 0 and j_start == 0:
            i_end = self.patch_size + 2 * self.rf
            j_end = self.patch_size + 2 * self.rf
        # top
        elif i_start == 0:
            i_end = self.patch_size + 2 * self.rf
            j_end = j_start + self.patch_size + self.rf
            j_start -= self.rf
        elif j_start == 0:
            j_end = self.patch_size + 2 * self.rf
            i_end = i_start + self.patch_size + self.rf
            i_start -= self.rf
        else:
            i_end = i_start + self.patch_size + self.rf
            i_start -= self.rf
            j_end = j_start + self.patch_size + self.rf
            j_start -= self.rf

        img = self._get_sentinel_data(i_start, i_end, j_start, j_end)
        img, _ = self.transform((img, np.empty((1, 1, 1))))
        patch = {
            'x': img,
            'row': (i_start, i_end),
            'col': (j_start, j_end)
        }

        return patch

    def _get_sentinel_data(self, i_start: int, i_end: int, j_start: int, j_end: int):
        img_patch = self.img[i_start:i_end, j_start:j_end, ]
        return np.nan_to_num(img_patch).astype(np.float32)

    @ staticmethod
    def _get_feature_selection(features, selection):
        feature_selection = [False for _ in range(len(features))]
        for feature in selection:
            i = features.index(feature)
            feature_selection[i] = True
        return feature_selection

    def get_mask(self, data_type = 'uint8') -> np.ndarray:
        mask = np.empty(shape=(self.n_rows * self.patch_size, self.n_cols * self.patch_size, 1), dtype=data_type)
        return mask

    def __len__(self):
        return self.length


# dataset for classifying a scene
class TilesInferenceDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, site: str, root_dir: Path = None):
        super().__init__()

        self.cfg = cfg
        self.site = site
        self.root_dir = Path(cfg.DATASETS.PATH) if root_dir is None else root_dir
        self.transform = transforms.Compose([Numpy2Torch()])

        # getting all files
        samples_file = self.root_dir / site / 'samples.json'
        metadata = load_json(samples_file)
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
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_sentinel2_data(self, patch_id):
        file = self.root_dir / self.site / 'sentinel2' / f'sentinel2_{self.site}_{patch_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_label_data(self, patch_id):
        label = self.cfg.DATALOADER.LABEL
        threshold = self.cfg.DATALOADER.LABEL_THRESH

        label_file = self.root_dir / self.site / label / f'{label}_{self.site}_{patch_id}.tif'
        img, transform, crs = read_tif(label_file)
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


class GHSLDataset(torch.utils.data.Dataset):

    def __init__(self, root_path: Path, site: str, label_thresh: float = None):
        super().__init__()

        self.root_path = root_path
        self.site = site
        self.thresh = label_thresh
        self.transform = transforms.Compose([Numpy2Torch()])

        # getting all files
        samples_file = self.root_path / site / 'samples.json'
        metadata = load_json(samples_file)
        self.samples = metadata['samples']
        self.length = len(self.samples)
        self.patch_size = metadata['patch_size']
        self.max_y = metadata['max_y']
        self.max_x = metadata['max_x']

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]
        patch_id = sample['patch_id']

        ghsl, _, _ = self._get_ghsl(patch_id)
        label, _, _ = self._get_label('buildings', patch_id)

        ghsl, label = self.transform((ghsl, label))

        item = {
            'x': ghsl,
            'y': label,
            'site': self.site,
            'patch_id': patch_id,
        }

        return item

    def _get_label(self, label_name: str, patch_id: str):

        label_file = self.root_path / self.site / label_name / f'{label_name}_{self.site}_{patch_id}.tif'
        img, transform, crs = read_tif(label_file)
        if self.thresh is not None:
            img = img > self.thresh

        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_ghsl(self, patch_id: str):

        ghsl_file = self.root_path / self.site / 'ghsl' / f'ghsl_{self.site}_{patch_id}.tif'
        img, transform, crs = read_tif(ghsl_file)
        img = img / 100

        return np.nan_to_num(img).astype(np.float32), transform, crs

    def get_arr(self, dtype=np.uint8):
        height = self.max_y + self.patch_size
        width = self.max_x + self.patch_size
        return np.zeros((height, width, 1), dtype=dtype)

    def get_geo(self):
        patch_id = f'{0:010d}-{0:010d}'
        _, transform, crs = self._get_data('ghsl', patch_id, None)
        return transform, crs

    def __len__(self):
        return self.length


# dataset for classifying a scene
class StockholmTimeseriesDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, root_dir: Path, date: str):
        super().__init__()

        self.cfg = cfg
        self.root_dir = root_dir
        self.date = date
        self.transform = transforms.Compose([Numpy2Torch()])

        # getting all files
        samples_file = self.root_dir / f'samples_{date}.json'
        metadata = load_json(samples_file)
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

        dummy_label = np.zeros((self.patch_size, self.patch_size, 1), dtype=np.float32)
        label = dummy_label
        extended_patch, label = self.transform((extended_patch, label))

        transform, crs = self._get_geo_patch(patch_id_center)

        item = {
            'x': extended_patch,
            'i': y_center,
            'j': x_center,
            'patch_id': patch_id_center,
            'transform': transform,
            'crs': crs
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
        file = self.root_dir / self.date / 'sentinel1' / f'sentinel1_stockholm_{self.date}_{patch_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_sentinel2_data(self, patch_id):
        file = self.root_dir / self.date / 'sentinel2' / f'sentinel2_stockholm_{self.date}_{patch_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_geo_patch(self, patch_id):
        _, transform, crs = self._get_sentinel1_data(patch_id)
        return transform, crs

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

