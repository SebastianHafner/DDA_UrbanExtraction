from pathlib import Path
from utils.geotiff import *
import json
import utm
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.visualization import *


SN7_PATH = Path('/storage/shafner/spacenet7/train')
# SN7_PATH = Path('C:/Users/hafne/urban_extraction/data/spacenet7/train')
METADATA_FILE = SN7_PATH.parent / 'sn7_metadata_urban_dataset.csv'


def get_all_aoi_ids() -> list:
    return [f.name for f in SN7_PATH.iterdir() if f.is_dir()]


def fname2date(fname: str) -> tuple:
    fname_parts = fname.split('_')
    year = int(fname_parts[2])
    month = int(fname_parts[3])
    return year, month


def fname2id(fname: str) -> tuple:
    fname_parts = fname.split('_')
    aoi_id = fname_parts[5]
    return aoi_id


def get_available_dates(aoi_id: str) -> list:
    images_path = SN7_PATH / aoi_id / 'images'
    dates = [fname2date(f.name) for f in images_path.glob('**/*')]
    return dates


def create_base_name(aoi_id: str, year: int, month: int) -> str:
    return f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}'


def get_shape(aoi_id: str) -> tuple:
    images_path = SN7_PATH / aoi_id / 'images'
    image_file = [f for f in images_path.glob('**/*')][0]
    arr, _, _ = read_tif(image_file)
    return arr.shape[0], arr.shape[1]


def get_geo(aoi_id: str):
    images_path = SN7_PATH / aoi_id / 'images'
    image_file = [f for f in images_path.glob('**/*')][0]
    _, transform, crs = read_tif(image_file)
    return transform, crs


# converts list of geojson polygons in pixel coordinates to raster image
def polygons2raster(polygons: list, shape: tuple = (1024, 1024)) -> np.ndarray:
    raster = np.zeros(shape)

    for polygon in polygons:
        # list of polygon elements: first element is the polygon outline and others are holes
        polygon_elements = polygon['geometry']['coordinates']

        # filling in the whole polygon
        polygon_outline = polygon_elements[0]
        first_coord = polygon_outline[0]
        # TODO: some coords are 3-d for some stupid reason, maybe fix?
        if len(first_coord) == 3:
            polygon_outline = [coord[:2] for coord in polygon_outline]
        polygon_outline = np.array(polygon_outline, dtype=np.int32)
        cv2.fillPoly(raster, [polygon_outline], 1)

        # setting holes in building back to 0
        # all building elements but the first one are considered holes
        if len(polygon_elements) > 1:
            for j in range(1, len(polygon_elements)):
                polygon_hole = polygon_elements[j]
                first_coord = polygon_hole[0]
                if len(first_coord) == 3:
                    polygon_hole = [coord[:2] for coord in polygon_hole]
                polygon_hole = np.array(polygon_hole, dtype=np.int32)
                cv2.fillPoly(raster, [polygon_hole], 0)

    return raster


def extract_bbox(aoi_id: str):

    root_path = SN7_PATH / aoi_id
    img_folder = root_path / 'images'
    all_img_files = list(img_folder.glob('**/*.tif'))
    img_file = all_img_files[0]
    arr, transform, crs = read_tif(img_file)
    y_pixels, x_pixels, _ = arr.shape

    x_pixel_spacing = transform[0]
    x_min = transform[2]
    x_max = x_min + x_pixels * x_pixel_spacing

    y_pixel_spacing = transform[4]
    y_max = transform[5]
    y_min = y_max + y_pixels * y_pixel_spacing

    bbox = ee.Geometry.Rectangle([x_min, y_min, x_max, y_max], proj=str(crs)).transform('EPSG:4326')
    return bbox


def epsg_utm(bbox):
    center_point = bbox.centroid()
    coords = center_point.getInfo()['coordinates']
    lon, lat = coords
    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
    return f'EPSG:326{zone_number}' if lat > 0 else f'EPSG:327{zone_number}'


def building_footprint_features(aoi_id, year, month):
    root_path = SN7_PATH / aoi_id
    label_folder = root_path / 'labels_match'
    label_file = label_folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.geojson'

    with open(str(label_file)) as f:
        label_data = json.load(f)

    features = label_data['features']
    new_features = []
    for feature in features:
        coords = feature['geometry']['coordinates']
        geom = ee.Geometry.Polygon(coords, proj='EPSG:3857').transform('EPSG:4326')
        new_feature = ee.Feature(geom)
        new_features.append(new_feature)
    return new_features


def construct_buildings_file(metadata_file: Path):

    metadata = pd.read_csv(metadata_file)

    merged_buildings = None
    for index, row in metadata.iterrows():
        aoi_id, year, month = row['aoi_id'], row['year'], row['month']
        file_name = f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.geojson'
        file = SN7_PATH / aoi_id / 'labels_match' / file_name
        with open(str(file)) as f:
            buildings = json.load(f)

        if merged_buildings is None:
            merged_buildings = buildings
        else:
            merged_features = merged_buildings['features']
            merged_features.extend(buildings['features'])
            merged_buildings['features'] = merged_features

    buildings_file = SN7_PATH.parent / f'sn7_buildings.geojson'
    with open(str(buildings_file), 'w', encoding='utf-8') as f:
        json.dump(merged_buildings, f, ensure_ascii=False, indent=4)


def construct_samples_file(metadata_file: Path, save_path: Path):
    metadata = pd.read_csv(metadata_file)
    samples = []
    for index, row in metadata.iterrows():
        sample = {
            'aoi_id': str(row['aoi_id']),
            'group': int(row['group']),
            'country': str(row['country']),
            'month': int(row['month']),
            'year': int(row['year']),
        }
        samples.append(sample)

    # writing data to json file
    data = {
        'label': 'buildings',
        'sentinel1_features': ['VV', 'VH'],
        'sentinel2_features': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
        'group_names': {'1': 'NA_AU', '2': 'SA', '3': 'EU', '4': 'SSA', '5': 'NAF_ME', '6': 'AS'},
        'samples': samples
    }
    dataset_file = save_path / f'samples.json'
    with open(str(dataset_file), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def dt_months(year: int, month: int, dt_months: int):
    if month > dt_months:
        return year, month - dt_months
    else:
        return year - 1, 12 - abs(month - dt_months)


def find_offset_file(aoi_id: str, year: int, month: int):

    ref_year, ref_month = dt_months(year, month, 6)

    while True:
        ref_file_name = f'global_monthly_{ref_year}_{ref_month:02d}_mosaic_{aoi_id}_Buildings.geojson'
        ref_file = SN7_PATH / aoi_id / 'labels_match' / ref_file_name
        if ref_file.exists():
            return ref_file
        else:
            ref_year, ref_month = dt_months(ref_year, ref_month, 1)
        if year < 2018:
            raise Exception('No file found!')


def construct_stable_buildings_file(metadata_file: Path):

    metadata = pd.read_csv(metadata_file)

    merged_buildings = None
    for index, row in metadata.iterrows():
        aoi_id, year, month = row['aoi_id'], row['year'], row['month']
        file_name = f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.geojson'
        file = SN7_PATH / aoi_id / 'labels_match' / file_name

        ref_file = find_offset_file(aoi_id, year, month)

        buildings = load_json(file)
        ref_buildings = load_json(ref_file)

        ref_ids = [feature['properties']['Id'] for feature in ref_buildings['features']]

        stable_features = []
        for feature in buildings['features']:
            feature_id = feature['properties']['Id']
            stable = 1 if feature_id in ref_ids else 0
            feature['properties']['stable'] = stable
            stable_features.append(feature)

        if merged_buildings is None:
            buildings['features'] = []
            merged_buildings = buildings

        merged_features = merged_buildings['features']
        merged_features.extend(stable_features)
        merged_buildings['features'] = merged_features

    # buildings_file = SPACENET7_PATH.parent / f'sn7_buildings.geojson'
    # with open(str(buildings_file), 'w', encoding='utf-8') as f:
    #     json.dump(merged_buildings, f, ensure_ascii=False, indent=4)


def construct_reference_buildings_file(metadata_file: Path):
    metadata = pd.read_csv(metadata_file)

    merged_buildings = None
    for index, row in metadata.iterrows():
        aoi_id, year, month = row['aoi_id'], row['year'], row['month']

        ref_file = find_offset_file(aoi_id, year, month)
        buildings = load_json(ref_file)

        if merged_buildings is None:
            merged_buildings = buildings
        else:
            merged_features = merged_buildings['features']
            merged_features.extend(buildings['features'])
            merged_buildings['features'] = merged_features

    buildings_file = SN7_PATH.parent / f'sn7_reference_buildings.geojson'
    with open(str(buildings_file), 'w', encoding='utf-8') as f:
        json.dump(merged_buildings, f, ensure_ascii=False, indent=4)


def create_label_masks(aoi_id):

    masks_path = SN7_PATH / aoi_id / 'masks'
    masks_path.mkdir(exist_ok=True)

    shape = get_shape(aoi_id)

    dates = get_available_dates(aoi_id)
    for year, month in dates:
        base_name = create_base_name(aoi_id, year, month)

        image_file = SN7_PATH / aoi_id / 'images' / f'{base_name}.tif'
        _, transform, crs = read_tif(image_file)

        buildings_file = SN7_PATH / aoi_id / 'labels_match_pix' / f'{base_name}_Buildings.geojson'
        buildings_data = load_json(buildings_file)
        buildings = buildings_data['features']
        mask = polygons2raster(buildings, shape=shape)

        udm_file = SN7_PATH / aoi_id / 'UDM_masks' / f'{base_name}_UDM.tif'
        if udm_file.exists():
            udm, _, _ = read_tif(udm_file)
            mask = np.clip(mask + (udm.squeeze() / 255 * 2), 0, 2)


        output_file = masks_path / f'{base_name}_mask.tif'
        write_tif(output_file, np.expand_dims(mask, -1), transform, crs)


# requires label masks
def show_stable_building_pixels(aoi_id: str):
    mask_path = SN7_PATH / aoi_id / 'masks'
    mask_files = [f for f in mask_path.glob('**/*')]

    shape = (*get_shape(aoi_id), len(mask_files))
    all_masks = np.zeros(shape)
    for i, file in enumerate(mask_files):
        mask, _, _ = read_tif(file)
        all_masks[:, :, i] = mask.squeeze()

    # TODO: could be all masked
    stable_buildings = np.all(all_masks, axis=-1).astype('uint8')

    # non stable buildings
    only_buildings = all_masks.copy()
    only_buildings[only_buildings == 2] = 0
    all_buildings = np.any(only_buildings, axis=-1).astype('uint8')
    img = all_buildings + stable_buildings

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(all_buildings, interpolation='nearest')
    axs[1].imshow(stable_buildings, interpolation='nearest')
    plot_stable_buildings_v2(axs[2], img)
    plt.suptitle(aoi_id)
    for ax in axs:
        ax.set_axis_off()
    plt.axis('off')
    plt.show()


def create_building_change_masks(aoi_id: str):
    metadata = pd.read_csv(METADATA_FILE)
    row = metadata[metadata['aoi_id'] == aoi_id]
    ref_year = int(row['year'])
    ref_month = int(row['month'])
    ref_date = ref_year * 12 + ref_month

    period_months = 12
    offset = period_months // 2

    # get dates of images within 1 year period
    dates = get_available_dates(aoi_id)
    dates = [d for d in dates if ref_date - offset <= (d[0] * 12 + d[1]) < ref_date + offset]

    mask_path = SN7_PATH / aoi_id / 'masks'
    shape = (*get_shape(aoi_id), len(dates))
    masks = np.zeros(shape)

    for i, date in enumerate(dates):
        year, month = date
        base_name = create_base_name(aoi_id, year, month)
        file = mask_path / f'{base_name}_mask.tif'
        mask, _, _ = read_tif(file)
        masks[:, :, i] = mask.squeeze()

    # TODO: could be all masked
    stable_buildings = np.all(masks, axis=-1).astype('uint8')

    # non stable buildings
    only_buildings = masks.copy()
    only_buildings[only_buildings == 2] = 0
    all_buildings = np.any(only_buildings, axis=-1).astype('uint8')

    change = all_buildings + stable_buildings
    output_file = SN7_PATH / aoi_id / 'auxiliary' / f'change.tif'
    transform, crs = get_geo(aoi_id)
    write_tif(output_file, np.expand_dims(change, axis=-1), transform, crs)


if __name__ == '__main__':

    # ee.Initialize()

    test_aoi = 'L15-0571E-1075N_2287_3888_13'

    # for aoi_id in tqdm(get_all_aoi_ids()):
        # show_stable_building_pixels(aoi_id)
        # create_label_masks(aoi_id)
        # create_building_change_masks(aoi_id)
    # show_stable_building_pixels(test_aoi)
    # construct_reference_buildings_file(metadata_file)

    # construct_buildings_file(METADATA_FILE)

    samples_save_path = Path('/storage/shafner/urban_extraction/urban_dataset/sn7')
    construct_samples_file(METADATA_FILE, samples_save_path)
