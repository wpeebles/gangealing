import argparse
import multiprocessing
from functools import partial
from io import BytesIO

import lmdb
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import cv2
import sys
import json
import os

from glob import glob
from utils.CUB_data_utils import square_bbox, perturb_bbox, acsm_crop
from utils.download import download_spair, download_lsun, download_cub, download_cub_metadata


# When an image is mirrored, any key points with left/right distinction need to be swapped.
# These are the permutations of key point indices that accomplishes this:
CUB_PERMUTATION = [0, 1, 2, 3, 4, 5, 10, 11, 12, 9, 6, 7, 8, 13, 14]
SPAIR_PERMUTATIONS = {
    'bicycle': [0, 1, 3, 2, 4, 5, 7, 6, 8, 10, 9, 11],
    'cat': [1, 0, 3, 2, 5, 4, 7, 6, 8, 10, 9, 12, 11, 13, 14],
    'dog': [1, 0, 3, 2, 5, 4, 6, 7, 8, 10, 9, 12, 11, 13, 14, 15],
    'tvmonitor': [2, 1, 0, 7, 6, 5, 4, 3, 10, 9, 8, 15, 14, 13, 12, 11],
}


def black_bar_pad(img, target_res, resize=True, to_pil=True):
    canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
    original_width, original_height = img.size
    if original_height <= original_width:
        if resize:
            img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.ANTIALIAS)
        width, height = img.size
        img = np.asarray(img)
        canvas[(width - height) // 2: (width + height) // 2] = img
    else:
        if resize:
            img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.ANTIALIAS)
        width, height = img.size
        img = np.asarray(img)
        canvas[:, (height - width) // 2: (height + width) // 2] = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas


def border_pad(img, target_res, resize=True, to_pil=True):
    original_width, original_height = img.size
    if original_height <= original_width:
        if resize:
            img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.ANTIALIAS)
        width, height = img.size
        img = np.asarray(img)
        half_height = (target_res - height) / 2
        int_half_height = int(half_height)
        lh = int_half_height
        rh = int_half_height + (half_height > int_half_height)
        img = np.pad(img, mode='edge', pad_width=[(lh, rh), (0, 0), (0, 0)])
    else:
        if resize:
            img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.ANTIALIAS)
        width, height = img.size
        img = np.asarray(img)
        half_width = (target_res - width) / 2
        int_half_width = int(half_width)
        lw = int_half_width
        rw = int_half_width + (half_width > int_half_width)
        img = np.pad(img, mode='edge', pad_width=[(0, 0), (lw, rw), (0, 0)])
    if to_pil:
        img = Image.fromarray(img)
    return img


def center_crop(img, target_res):
    # From official StyleGAN2 create_lsun method:
    img = np.asarray(img)
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    img = Image.fromarray(img, 'RGB')
    img = img.resize((target_res, target_res), Image.ANTIALIAS)
    return img


def cub_crop(img, target_res, bbox):
    # This function mimics ACSM's pre-processing used for the CUB dataset (up to image resampling and padding color)
    img = np.asarray(img)
    img = acsm_crop(img, bbox, 0, border=True)
    return Image.fromarray(img).resize((target_res, target_res), Image.ANTIALIAS)


def preprocess_kps_pad(kps, img_width, img_height, size):
    # Once an image has been pre-processed via border (or zero) padding,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the border-padded (or zero-padded) image.
    kps = kps.clone()
    scale = size / max(img_width, img_height)
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        offset_y = int((size - new_h) / 2)
        offset_x = 0
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        offset_x = int((size - new_w) / 2)
        offset_y = 0
        kps[:, 0] += offset_x
    else:
        offset_x = 0
        offset_y = 0
    kps *= kps[:, 2:3]  # zero-out any non-visible key points
    return kps, offset_x, offset_y, scale


def preprocess_kps_box_crop(kps, bbox, size):
    # Once an image has been pre-processed via a box crop,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the cropped image.
    kps = kps.clone()
    kps[:, 0] -= bbox[0] + 1
    kps[:, 1] -= bbox[1] + 1
    w = 1 + bbox[2] - bbox[0]
    h = 1 + bbox[3] - bbox[1]
    assert w == h
    kps[:, [0, 1]] *= size / float(w)
    return kps


def load_CUB_keypoints(path):
    names = ['img_index', 'kp_index', 'x', 'y', 'visible']
    landmarks = pd.read_table(path, header=None, names=names, delim_whitespace=True, engine='python')
    landmarks = landmarks.to_numpy().reshape((11788, 15, 5))[..., [2, 3, 4]]  # (num_images, num_kps, 3)
    landmarks = torch.from_numpy(landmarks).float()
    return landmarks


def load_acsm_data(path, mat_path='data/val_cub_cleaned_new.mat', size=256, out_path=None):
    from scipy.io import loadmat
    mat = loadmat(mat_path)
    files = [f'data/CUB_200_2011/images/{file[0]}' for file in mat['images']['rel_path'][0]]
    # These are the indices retained by ACSM (others are filtered):
    indices = [i[0, 0] - 1 for i in mat['images']['id'][0]]
    kps = load_CUB_keypoints(f'{path}/parts/part_locs.txt')[indices]
    b = mat['images']['bbox'][0]
    bboxes = []
    kps_out = []
    for ix, row in enumerate(b):
        x1, y1, x2, y2 = row[0, 0]
        bbox = np.array([x1[0, 0], y1[0, 0], x2[0, 0], y2[0, 0]]) - 1
        bbox = perturb_bbox(bbox, 0.05, 0)
        bbox = square_bbox(bbox)
        bboxes.append(bbox)
        kps_out.append(preprocess_kps_box_crop(kps[ix], bbox, size))
    bboxes = np.stack(bboxes)
    kps_out = torch.stack(kps_out)
    torch.save(kps_out, f'{out_path}/keypoints.pt')
    # When an image is mirrored horizontally, the designation between key points with a left versus right distinction
    # needs to be swapped. This is the permutation of CUB key points which accomplishes this swap:
    torch.save(CUB_PERMUTATION, f'{out_path}/permutation.pt')
    assert bboxes.shape[0] == len(files)
    return files, bboxes


def load_spair_data(path, size, out_path, category='cat', split='test'):
    pairs = sorted(glob(f'{path}/PairAnnotation/{split}/*:{category}.json'))
    files = []
    thresholds = []
    inverse = []
    category_anno = list(glob(f'{path}/ImageAnnotation/{category}/*.json'))[0]
    with open(category_anno) as f:
        num_kps = len(json.load(f)['kps'])
    print(f'Number of SPair key points for {category} <= {num_kps}')
    kps = []
    blank_kps = torch.zeros(num_kps, 3)
    for pair in pairs:
        with open(pair) as f:
            data = json.load(f)
        assert category == data["category"]
        assert data["mirror"] == 0
        source_fn = f'{path}/JPEGImages/{category}/{data["src_imname"]}'
        target_fn = f'{path}/JPEGImages/{category}/{data["trg_imname"]}'
        source_bbox = np.asarray(data["src_bndbox"])
        target_bbox = np.asarray(data["trg_bndbox"])
        # The source thresholds aren't actually used to evaluate PCK on SPair-71K, but for completeness
        # they are computed as well:
        thresholds.append(max(source_bbox[3] - source_bbox[1], source_bbox[2] - source_bbox[0]))
        thresholds.append(max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0]))

        source_size = data["src_imsize"][:2]  # (W, H)
        target_size = data["trg_imsize"][:2]  # (W, H)

        kp_ixs = torch.tensor([int(id) for id in data["kps_ids"]]).view(-1, 1).repeat(1, 3)
        source_raw_kps = torch.cat([torch.tensor(data["src_kps"], dtype=torch.float), torch.ones(kp_ixs.size(0), 1)], 1)
        source_kps = blank_kps.scatter(dim=0, index=kp_ixs, src=source_raw_kps)
        source_kps, src_x, src_y, src_scale = preprocess_kps_pad(source_kps, source_size[0], source_size[1], size)
        target_raw_kps = torch.cat([torch.tensor(data["trg_kps"], dtype=torch.float), torch.ones(kp_ixs.size(0), 1)], 1)
        target_kps = blank_kps.scatter(dim=0, index=kp_ixs, src=target_raw_kps)
        target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(target_kps, target_size[0], target_size[1], size)

        kps.append(source_kps)
        kps.append(target_kps)
        files.append(source_fn)
        files.append(target_fn)
        inverse.append([src_x, src_y, src_scale])
        inverse.append([trg_x, trg_y, trg_scale])
    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]
    print(f'Final number of used key points: {kps.size(1)}')
    num_imgs = len(thresholds)  # Total number of images (= 2 * number of pairs)
    torch.save(torch.arange(num_imgs).view(num_imgs // 2, 2), f'{out_path}/pairs.pt')
    torch.save(torch.tensor(thresholds, dtype=torch.float), f'{out_path}/pck_thresholds.pt')
    torch.save(torch.tensor(inverse), f'{out_path}/inverse_coordinates.pt')
    torch.save(kps, f'{out_path}/keypoints.pt')
    torch.save(SPAIR_PERMUTATIONS[category], f'{out_path}/permutation.pt')
    return files, [None] * len(files)  # No bounding boxes are used


def load_image_folder(path, pattern):
    files = sorted(glob(f'{path}/{pattern}'))
    bboxes = [None] * len(files)  # This means no bounding boxes are used
    return files, bboxes


def resize_and_convert(img, size, pad, quality=100, format='jpeg', bbox=None):
    if pad == 'zero':
        img = black_bar_pad(img, size)
    elif pad == 'border':
        img = border_pad(img, size)
    elif pad == 'center':
        img = center_crop(img, size)
    elif pad == 'none':
        pass
    elif pad == 'cub_crop':
        img = cub_crop(img, size, bbox)
    else:
        raise NotImplementedError
    # img = trans_fn.resize(img, size, resample)
    # img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format=format, quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(img, sizes=(128, 256, 512, 1024), quality=100, pad='zero', format='jpeg', bbox=None):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, pad, quality, format, bbox))

    return imgs


def resize_worker(img_file, sizes, pad, format, lmdb_path):
    i, file, bbox = img_file
    if lmdb_path is not None:  # Load image from LMDB (useful for LSUN datasets, etc.)
        input_env = lmdb.open(lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        with input_env.begin(write=False) as txn:
            img_bytes = txn.get(file)
        try:  # https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/dataset_tool.py
            try:
                img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), 1)
                if img is None:
                    raise IOError('cv2.imdecode failed')
                img = img[:, :, ::-1]  # BGR => RGB
                img = Image.fromarray(img)
            except IOError:
                img = Image.open(BytesIO(img_bytes))
            out = resize_multiple(img, sizes=sizes, pad=pad, format=format, bbox=bbox)
        except:  # Corrupted image that should be skipped
            out = [None] * len(sizes)
            print(sys.exc_info()[1])
    else:  # Directly open a saved image:
        img = Image.open(file).convert('RGB')
        out = resize_multiple(img, sizes=sizes, pad=pad, format=format, bbox=bbox)
    return i, out


def prepare(
    env, path, out, n_worker, sizes=(128, 256, 512, 1024), pad='zero', format='jpeg', input_is_lmdb=False,
        pattern='*.png', max_images=None, spair_category=None, spair_split=None, cub_acsm=False
):
    if input_is_lmdb:
        lmdb_path = path
        input_env = lmdb.open(lmdb_path, readonly=True, lock=False)
        print('Loading LMDB keys (this might take a bit)...')
        with input_env.begin(write=False) as inp_txn:
            key_list = list(inp_txn.cursor().iternext(values=False))  # https://stackoverflow.com/a/65663873
        if max_images is not None:
            key_list = key_list[:max_images]
        num_files = len(key_list)
        print(f'LMDB keys loaded! Found {num_files} keys.')
        files = [(i, key, None) for i, key in enumerate(key_list)]
    else:
        lmdb_path = None
        if cub_acsm:  # Load CUB using ACSM pre-processing (this is the only dataset that uses bboxes in pre-processing)
            files, bboxes = load_acsm_data(path, size=int(sizes[0]), out_path=out)
        elif spair_category is not None:  # Load SPair-71K (bboxes = None)
            files, bboxes = load_spair_data(path, size=int(sizes[0]), out_path=out,
                                            category=spair_category, split=spair_split)
        else:  # Load images from a folder (or hierarchy of folders); bboxes = None
            files, bboxes = load_image_folder(path, pattern)
        if max_images is not None:
            files = files[:max_images]
            bboxes = bboxes[:max_images]
        num_files = len(files)
        print(f'Found {num_files} files')
        print(f'Example file being loaded: {files[0]}')
        files = [(i, file, bbox) for i, (file, bbox) in enumerate(zip(files, bboxes))]

    resize_fn = partial(resize_worker, sizes=sizes, pad=pad, format=format, lmdb_path=lmdb_path)
    total = 0
    skipped = 0
    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap(resize_fn, files), total=num_files):
            increment = 1
            for size, img in zip(sizes, imgs):
                if img is None:
                    print('skipping image')
                    increment = 0
                    skipped += 1
                else:
                    key = f"{size}-{str(i - skipped).zfill(5)}".encode("utf-8")

                    with env.begin(write=True) as txn:
                        txn.put(key, img)

            total += increment

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))
    print(f'Final dataset size: {total}')


def create_dataset(out, path, size, pad='zero', n_worker=8, format='jpeg', input_is_lmdb=False, pattern='*.png',
                   max_images=None, spair_category=None, spair_split=None, cub_acsm=False):
    size = str(size)

    sizes = [int(s.strip()) for s in size.split(",")]

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    with lmdb.open(out, map_size=2048 ** 4, readahead=False) as env:
        prepare(env, path, out, n_worker, sizes=sizes, pad=pad, format=format,
                input_is_lmdb=input_is_lmdb, pattern=pattern, max_images=max_images,
                spair_category=spair_category, spair_split=spair_split, cub_acsm=cub_acsm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create image datasets")
    parser.add_argument("--out", type=str, help="filename of the output lmdb dataset")
    parser.add_argument(
        "--size",
        type=str,
        default="256",
        help="resolutions of images for the dataset",
    )
    parser.add_argument(
        "--n_worker",
        type=int,
        default=8,
        help="number of workers for preparing dataset",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['png', 'jpeg'],
        default='png',
        help="format to store images in the database",
    )
    parser.add_argument("--pad", choices=['zero', 'border', 'center', 'none'], default='center')
    parser.add_argument("--path", type=str, help="path to the image dataset", default=None)
    parser.add_argument("--input_is_lmdb", action='store_true',
                        help='If true, path input points to an LMDB dataset. This is useful for, e.g., creating '
                             'LSUN datasets. If you use this you can ignore --pattern')
    parser.add_argument("--pattern", type=str, default='*.png', help='Specify the pattern glob uses to find images')
    parser.add_argument("--max_images", type=int, default=None, help='Maximum number of images to include in '
                                                                     'final dataset (default: include all)')

    # Special arguments for loading SPair-71K and CUB for PCK evaluation purposes (and also LSUN). If you use these
    # options below, you can ignore --input_is_lmdb, --path and --pattern above.
    parser.add_argument("--spair_category", default=None, type=str, choices=list(SPAIR_PERMUTATIONS.keys()),
                        help='If specified, constructs the SPair-71K dataset for the specified category')
    parser.add_argument("--spair_split", default='test', choices=['trn', 'val', 'test'], type=str,
                        help='The split of SPair that will be constructed (only used if --spair_category is specified)')
    parser.add_argument("--lsun_category", default=None, type=str,
                        help='If specified, constructs the LSUN dataset for the specified category '
                             '(may take a while to download!)')
    parser.add_argument("--cub_acsm", action='store_true',
                        help='If true, constructs the CUB dataset. This will use the same pre-processing and filtering '
                             'as the CUB validation split from the ACSM paper.')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Automatically download some datasets:
    if args.cub_acsm:  # Download metadata for CUB pre-processing
        os.makedirs('data', exist_ok=True)
        download_cub_metadata('data')
        args.pad = 'cub_crop'
    elif args.spair_category is not None:
        args.pad = 'border'  # zero padding would also be acceptable
    if args.cub_acsm and args.path is None:  # Download CUB-200-2011 if needed
        args.path = download_cub('data')
    elif args.spair_category is not None and args.path is None:  # Download SPair-71K data if needed
        os.makedirs('data', exist_ok=True)
        args.path = download_spair('data')
        args.pad = 'border'
    elif args.lsun_category is not None:  # Download LSUN category automatically
        os.makedirs('data', exist_ok=True)
        args.path = download_lsun('data', args.lsun_category)
        args.input_is_lmdb = True
    else:
        assert args.path is not None

    create_dataset(args.out, args.path, args.size, args.pad, args.n_worker, args.format,
                   args.input_is_lmdb, args.pattern, args.max_images, args.spair_category, args.spair_split,
                   args.cub_acsm)
