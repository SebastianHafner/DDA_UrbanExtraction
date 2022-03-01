import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import geofiles, parsers
from tqdm import tqdm


def patches2png(site: str, product: str, dataset_path: str, output_path: str):

    # loading metadata and unpacking
    folder = Path(dataset_path) / site
    metadata = geofiles.load_json(folder / 'samples.json')
    patches, patch_size = metadata['samples'], metadata['patch_size']
    max_x, max_y = metadata['max_x'], metadata['max_y']

    # creating container for img
    arr = np.zeros((max_y + patch_size, max_x + patch_size, 3))

    # filling img
    for index, patch in enumerate(tqdm(patches)):

        patch_id = patch['patch_id']
        patch_file = folder / product / f'{product}_{site}_{patch_id}.tif'

        patch_data, _, _ = geofiles.read_tif(patch_file)
        y, x = geofiles.id2yx(patch_id)

        if product == 'sentinel2':
            arr[y:y+patch_size, x:x+patch_size, ] = np.clip(patch_data[:, :, [2, 1, 0]] / 0.3, 0, 1)
        else:
            for i in range(3):
                arr[y:y + patch_size, x:x + patch_size, i] = patch_data[:, :, 0]

    plt.imshow(arr)
    plt.axis('off')
    save_file = Path(output_path) / f'{site}_{product}.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight')




if __name__ == '__main__':
    args = parsers.inspector_argument_parser().parse_known_args()[0]
    for site in args.sites:
        patches2png(site, args.product, args.dataset_dir, args.output_dir)
