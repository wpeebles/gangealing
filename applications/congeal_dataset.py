"""
This script takes a pre-trained Spatial Transformer and applies it to an unaligned dataset to create an aligned and
filtered dataset in an unsupervised fashion. By default, this script will only use the similarity transformation
portion of the Spatial Transformer (rotation + crop) to avoid introducing warping artifacts.
"""

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from prepare_data import create_dataset, border_pad
from models import ComposedSTN
from models.spatial_transformers.warping_heads import SimilarityHead
from applications import base_eval_argparse, load_stn, determine_flips
from applications.flow_scores import filter_dataset
from utils.distributed import setup_distributed, primary, get_rank, all_gatherv, synchronize, get_world_size
from datasets import MultiResolutionDataset
import os


def apply_congealing(args, dataset, stn, stn_full, out_path, device, rank, n_processes, **stn_args):

    def prepro(x, from_np=False):
        if from_np:
            x = np.asarray(x)
        return torch.from_numpy(x).float().div_(255.0).add_(-0.5).mul_(2.0).permute(2, 0, 1).unsqueeze_(0).to(device)

    total = 0
    prefix = chr(ord('a') + rank)
    print(f'({rank}) Using prefix {prefix}')
    pbar = tqdm if rank == 0 else lambda x: x
    indices = torch.arange(rank, len(dataset), n_processes)
    one_hot = torch.tensor([[[0, 0, 1]]], dtype=torch.float, device=device)
    used_indices = []
    for i in pbar(indices):
        with torch.no_grad():
            x = dataset[i.item()]  # (1, C, H, W)
            w, h = x.size
            size = max(w, h)
            x_big = prepro(border_pad(x, size, resize=False, to_pil=False))  # (1, C, size, size)
            x_in = prepro(border_pad(x, args.flow_size, to_pil=False))  # (1, C, flow_size, flow_size)
            x_in, flip_indices, warp_policy = determine_flips(args, stn_full, None, x_in)
            x_big = torch.where(flip_indices.view(-1, 1, 1, 1), x_big.flip(3,), x_big)
            image_bounds = torch.tensor([[h, w]], dtype=torch.float, device='cuda')
            try:
                aligned, M, oob = stn(x_in, return_flow=True, return_out_of_bounds=True, input_img_for_sampling=x_big,
                                      output_resolution=args.output_resolution, image_bounds=image_bounds, **stn_args)
            except RuntimeError:
                print(f'Rank {rank}: WARNING: Ran out of GPU memory, skipping...')
                continue
            # The scale of the similarity transform can be extracted from our affine matrix
            # by taking the square-root of its determinant:
            M = torch.cat([M, one_hot], 1)
            scale = torch.det(M).sqrt_()
            too_low_res = (scale.item() * min(w, h)) < args.min_effective_resolution
            # We don't want to include images that can only be aligned by extrapolating a significant number of pixels
            # beyond the image boundary:
            if not (too_low_res or oob.item()):
                used_indices.append(i)
                write_image_batch(aligned, out_path, start_index=total, prefix=prefix)
                total += aligned.size(0)
    print(f'({rank}) Saved {total} images')
    used_indices = torch.stack(used_indices).to(device)
    return used_indices


def write_image_batch(images, out_path, start_index=0, prefix=''):

    def norm(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min)

    norm(images, -1, 1)
    ndarr = images.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
    for i in range(ndarr.shape[0]):
        index = i + start_index
        Image.fromarray(ndarr[i]).save(f'{out_path}/{prefix}{index:07}.png')


def align_and_filter_dataset(args, t):
    # The aligned + filtered images will be saved directly as pngs to temp_folder below:
    temp_folder = f'{args.out}_imagefolder'
    if primary():
        os.makedirs(temp_folder, exist_ok=True)
        os.makedirs(args.out, exist_ok=True)

    # Step 1: Apply the STN to every image in the dataset
    dataset = MultiResolutionDataset(args.real_data_path, resolution=args.real_size, transform=None)
    if args.flow_scores is not None:  # Filter the dataset based on flow scores:
        dataset = filter_dataset(dataset, args.flow_scores, args.fraction_retained)
    if isinstance(t, ComposedSTN):
        t_sim = t.stns[0]  # Only use the similarity transformation
    else:
        t_sim = t
    assert isinstance(t_sim.warp_head, SimilarityHead), 'Currently only similarity transformations are supported ' \
                                                        'for this script'
    used_indices = apply_congealing(args, dataset, t_sim, t, temp_folder, 'cuda', get_rank(), get_world_size(),
                                    iters=args.iters, padding_mode=args.padding_mode)
    synchronize()
    used_indices = all_gatherv(used_indices)
    # Step 2: Create an lmdb from temp_folder:
    if primary():
        create_dataset(args.out, temp_folder, size=args.output_resolution, format='png')
        used_indices = used_indices.sort().values.cpu()
        print(f'Saving indices of images (size={used_indices.size(0)})')
        torch.save(used_indices, f'{args.out}/dataset_indices.pt')
        print('Done.')


if __name__ == '__main__':
    parser = base_eval_argparse()
    # Dataset congealing + creation hyperparameters:
    parser.add_argument("--out", type=str, help='Directory to save output aligned dataset', required=True)
    parser.add_argument("--output_resolution", type=int, default=256, help='Resolution of output aligned images')
    parser.add_argument("--flow_scores", default=None, type=str,
                        help='Path to pre-computed flow scores to filter dataset (see flow_scores.py for more info)')
    parser.add_argument("--fraction_retained", default=1.0, type=float,
                        help='Fraction of dataset images to retain based on flow scores')
    # Also see --fraction_retained in base_eval_argparse()
    parser.add_argument("--min_effective_resolution", type=int, default=192,
                        help='Some images will have small objects that the STN successfully aligns. But, you may not '
                             'want these aligned images in your dataset because the STN will have produced a large '
                             'zoom that yields a low resolution image when resized to output_resolution. Any aligned '
                             'image with size less than min_effective_resolution will be excluded from the output '
                             'dataset.')
    args = parser.parse_args()
    assert args.num_heads == 1, 'Clustering not currently supported for congeal_dataset.py'
    args.distributed = setup_distributed()
    t_ema = load_stn(args)
    align_and_filter_dataset(args, t_ema)
