from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
from utils import geofiles, parsers


# computing the percentage of urban pixels for a file
def get_image_weight(file: Path):
    if not file.exists():
        raise FileNotFoundError(f'Cannot find file {file.name}')
    arr, _, _ = geofiles.read_tif(file)
    n_urban = np.sum(arr)
    return int(n_urban)


def has_only_zeros(file: Path) -> bool:
    arr, _, _ = geofiles.read_tif(file)
    sum_ = np.sum(arr)
    if sum_ == 0:
        return True
    return False


def crop_patch(file: Path, patch_size: int):
    arr, transform, crs = geofiles.read_tif(file)
    i, j, _ = arr.shape
    if i > patch_size or j > patch_size:
        arr = arr[:patch_size, :patch_size, ]
        geofiles.write_tif(file, arr, transform, crs)
    elif i < patch_size or j < patch_size:
        raise Exception(f'invalid file found {file.name}')
    else:
        pass


def preprocess(path: Path, site: str, labels_exist: bool, patch_size: int = 256):

    print(f'preprocessing {site}')

    s1_path = path / site / 'sentinel1'
    patches = [f.stem.split('_')[-1] for f in s1_path.glob('**/*')]

    max_x, max_y = 0, 0

    samples = []
    for i, patch_id in enumerate(tqdm(patches)):

        sentinel1_file = path / site / 'sentinel1' / f'sentinel1_{site}_{patch_id}.tif'
        sentinel2_file = path / site / 'sentinel2' / f'sentinel2_{site}_{patch_id}.tif'
        files = [sentinel1_file, sentinel2_file]
        if labels_exist:
            buildings_file = path / site / 'buildings' / f'buildings_{site}_{patch_id}.tif'
            files.append(buildings_file)
            img_weight = get_image_weight(buildings_file)
        else:
            img_weight = -1

        valid = True
        for file in files:
            # if has_only_zeros(file):
            #     raise Exception(f'only zeros {file.name}')
            arr, transform, crs = geofiles.read_tif(file)
            m, n, _ = arr.shape
            if m != patch_size or n != patch_size:
                if m >= patch_size and n >= patch_size:
                    pass
                    # TODO: should be cropped if too large
                else:
                    valid = False

        sample = {
            'site': site,
            'patch_id': patch_id,
            'is_labeled': labels_exist,
            'img_weight': img_weight
        }
        if valid:
            samples.append(sample)
            y, x = geofiles.id2yx(patch_id)
            max_x = x if x > max_x else max_x
            max_y = y if y > max_y else max_y

    # writing data to json file
    data = {
        'label': 'buildings',
        'site': site,
        'patch_size': patch_size,
        'max_x': max_x,
        'max_y': max_y,
        'sentinel1_features': ['VV', 'VH'],
        'sentinel2_features': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
        'samples': samples
    }
    dataset_file = path / site / f'samples.json'
    with open(str(dataset_file), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    args = parsers.training_argument_parser().parse_known_args()[0]

    dataset_path = Path('C:/Users/shafner/datasets/urban_dataset')
    for site, labeled in zip(args.sites, args.labeled):
        preprocess(args.dataset_path, site, labels_exist=True if labeled == 'True' else False, patch_size=256)


