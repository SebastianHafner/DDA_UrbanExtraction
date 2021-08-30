import shutil, json
from pathlib import Path
from utils.geotiff import *
import numpy as np
from tqdm import tqdm


# computing the percentage of urban pixels for a file
def get_image_weight(file: Path):
    if not file.exists():
        raise FileNotFoundError(f'Cannot find file {file.name}')
    arr, _, _ = read_tif(file)
    n_urban = np.sum(arr)
    return int(n_urban)


def has_only_zeros(file: Path) -> bool:
    arr, _, _ = read_tif(file)
    sum_ = np.sum(arr)
    if sum_ == 0:
        return True
    return False


def crop_patch(file: Path, patch_size: int):
    arr, transform, crs = read_tif(file)
    i, j, _ = arr.shape
    if i > patch_size or j > patch_size:
        arr = arr[:patch_size, :patch_size, ]
        write_tif(file, arr, transform, crs)
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
            arr, transform, crs = read_tif(file)
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
            y, x = id2yx(patch_id)
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


def sites_split(sites: list, train_fraction: float):
    n = len(sites)
    n_train = int(n * train_fraction)
    print(n_train)
    training_sites = list(np.random.choice(sites, size=n_train, replace=False))
    validation_sites = [site for site in sites if site not in training_sites]
    print(training_sites, validation_sites)


if __name__ == '__main__':

    # dataset_path = Path('C:/Users/shafner/urban_extraction/data/dummy_data')
    dataset_path = Path('/storage/shafner/urban_extraction/urban_dataset')

    labeled_sites = ['albuquerque', 'atlanta', 'calgary', 'charlston', 'columbus', 'dallas', 'denver', 'elpaso',
                     'houston', 'kansascity', 'lasvegas', 'losangeles', 'miami', 'minneapolis', 'montreal', 'newyork',
                     'phoenix', 'quebec', 'saltlakecity', 'sandiego', 'sanfrancisco', 'santafe', 'seattle', 'stgeorge',
                     'toronto', 'tucson', 'vancouver', 'winnipeg']
    unlabeled_sites = ['beijing', 'dakar', 'dubai', 'jakarta', 'kairo', 'kigali', 'lagos', 'mexicocity', 'milano',
                       'mumbai', 'riodejanairo', 'shanghai', 'sidney', 'stockholm']
    unlabeled_sites = ['buenosaires', 'bogota', 'sanjose', 'santiagodechile', 'kapstadt', 'tripoli', 'freetown',
                       'london', 'madrid', 'kinshasa', 'manila', 'moscow', 'newdehli', 'nursultan', 'perth', 'tokio']
    unlabeled_sites = ['maputo', 'caracas', 'santacruzdelasierra', 'saopaulo', 'asuncion', 'lima', 'paramaribo',
                       'libreville', 'djibuti', 'beirut', 'baghdad', 'athens', 'islamabad', 'hanoi', 'bangkok',
                       'dhaka', 'bengaluru', 'taipeh', 'berlin', 'nanning', 'wuhan']

    labeled_sites = ['kampala', 'stockholm', 'daressalam', 'sidney']
    labeled_sites = []
    unlabeled_sites = ['charleston2016', 'charleston2020', 'daressalaam2016', 'daressalaam2020', 'detroit2016',
                       'detroit2020', 'guangzhou2016', 'guangzhou2020', 'heidelberg2016', 'heidelberg2020',
                       'lagos2016', 'lagos2020', 'lapaz2016', 'lapaz2020', 'mexicocity2016', 'mexicocity2020',
                       'mumbai2016', 'mumbai2020', 'nairobi2016', 'nairobi2020', 'newyork2016', 'newyork2020',
                       'nouakchott2016', 'nouakchott2020', 'shanghai2016', 'shanghai2020', 'sydney2016', 'sydney2020']
    unlabeled_sites = ['tehran', 'baghdad', 'tindouf', 'kuwaitcity', 'dahmar', 'sanaa']
    all_sites = labeled_sites + unlabeled_sites
    for i, site in enumerate(all_sites):
        if i >= 0:
            labeled = True if site in labeled_sites else False
            preprocess(dataset_path, site, labels_exist=labeled, patch_size=256)





