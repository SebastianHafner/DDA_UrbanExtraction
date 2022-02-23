def get_all_aoi_ids() -> list:
    dirs = paths.load_paths()
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
    arr, _, _ = geofiles.read_tif(image_file)
    return arr.shape[0], arr.shape[1]


def get_geo(aoi_id: str):
    images_path = SN7_PATH / aoi_id / 'images'
    image_file = [f for f in images_path.glob('**/*')][0]
    _, transform, crs = geofiles.read_tif(image_file)
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
    arr, transform, crs = geofiles.read_tif(img_file)
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