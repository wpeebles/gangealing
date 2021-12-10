"""
This script evaluates PCK-Transfer (this is an efficient implementation that fully supports DistributedDataParallel).
This can also be used to visualize key point transfers (--vis_transfer).
"""

import torch
import numpy as np
from applications import base_eval_argparse, load_stn
from datasets import pck_dataloader, sample_infinite_pck_data
from utils.distributed import setup_distributed, get_world_size, get_rank, primary, all_gather
from utils.vis_tools.helpers import batch_overlay, images2grid, save_image
from PIL import Image
import ray
import termcolor
from tqdm import tqdm


def run_pck_transfer(args, t):
    loader = pck_dataloader(args.real_data_path, resolution=args.real_size, seed=args.seed, batch_size=args.batch,
                            distributed=args.distributed, infinite=False)
    permutation = loader.dataset.mirror_permutation
    num_pairs = len(loader.dataset) if args.num_pck_pairs is None else args.num_pck_pairs
    match_flows = not args.no_flip_inference
    loader = sample_infinite_pck_data(loader)
    if primary() and args.vis_transfer:  # Optionally visualize some transfers:
        vis_transfer(t, loader, permutation, match_flows, args.out, iters=args.iters, padding_mode=args.padding_mode)
    # Compute PCK-Transfer:
    pck_alphas = pck_transfer(t, loader, args.alphas, quiet=False, permutation=permutation, num_pairs=num_pairs,
                              transfer_both_ways=args.transfer_both_ways, match_flows=match_flows,
                              iters=args.iters, padding_mode=args.padding_mode)
    if primary():
        pck_string = format_pck_string(pck_alphas, args.alphas)
        print(pck_string)


def run_pck_bootstrap(args, t):
    # This function runs PCK multiple times with re-sampled pairs to bootstrap error bars.
    # For datasets with fixed pairs (like SPair), we sample the fixed pairs with replacement
    loader = pck_dataloader(args.real_data_path, resolution=args.real_size, seed=args.seed, batch_size=args.batch,
                            distributed=args.distributed, infinite=False)
    permutation = loader.dataset.mirror_permutation
    num_pairs = len(loader.dataset) if args.num_pck_pairs is None else args.num_pck_pairs
    match_flows = not args.no_flip_inference
    rng = torch.Generator()
    rng.manual_seed(args.seed)  # This seed should be the same across GPUs for consistency
    BIG_NUMBER = 9999999999999
    pcks = []
    pbar = tqdm(range(args.num_bootstrap)) if primary() else range(args.num_bootstrap)
    for _ in pbar:
        if hasattr(loader.dataset, 'fixed_pairs'):
            pair_seed = torch.randint(0, BIG_NUMBER, (1,), generator=rng).item()
            loader.dataset.randomize_fixed_pairs(pair_seed)
        loader_in = sample_infinite_pck_data(loader)
        # Compute PCK-Transfer:
        pck_alphas = pck_transfer(t, loader_in, args.alphas, quiet=True, permutation=permutation, num_pairs=num_pairs,
                                  transfer_both_ways=args.transfer_both_ways, match_flows=match_flows,
                                  iters=args.iters, padding_mode=args.padding_mode)
        pcks.append(pck_alphas)
    if primary():
        stdevs = torch.stack(pcks, 0).std(dim=0, unbiased=True)
        bootstrap_string = format_pck_string(stdevs, args.alphas)
        print('-----Bootstrapping Results (standard deviations)-----')
        print(bootstrap_string)


def format_pck_string(pcks, alphas):
    pck_colors = ['blue', 'red', 'green', 'magenta', 'grey', 'cyan', 'white', 'yellow']
    pck_colors = pck_colors * (1 + len(pcks) // len(pck_colors))
    alpha2pck = zip(alphas, pcks)
    pck_str = ' | '.join(
        [termcolor.colored(f'PCK-Transfer@{alpha}: {np.round(pck_alpha.item() * 100, 2)}%', pck_colors[aix]) for
         aix, (alpha, pck_alpha) in enumerate(alpha2pck)])
    return pck_str


@torch.inference_mode()
def vis_transfer(t, loader, permutation, match_flows, out, num_to_vis=8, **stn_forward_kwargs):
    device = 'cuda'
    d = next(loader)
    imgsA, imgsB, gt_kpsA_original, gt_kpsB = d['imgsA'][:num_to_vis].to(device), d['imgsB'][:num_to_vis].to(device), \
                                              d['kpsA'][:num_to_vis, :, :2].to(device), d['kpsB'][:num_to_vis, :, :2].to(device)
    imgs = torch.cat([imgsA, imgsB]).cpu()  # Visualize the original images (before any flips are performed)
    if match_flows:
        imgsA, imgsB, gt_kpsA, gt_kpsB, indices = t.match_flows(imgsA, imgsB, gt_kpsA_original, gt_kpsB, permutation,
                                                                **stn_forward_kwargs)
    # Transfer the key points from imgsA to imgsB:
    est_kpsB = t.transfer_points(imgsA, imgsB, gt_kpsA, **stn_forward_kwargs)
    est_kpsB[:, :, 0] = torch.where(indices.view(imgsA.size(0), 1) > 1, imgsB.size(-1) - 1 - est_kpsB[:, :, 0], est_kpsB[:, :, 0])
    kps = torch.cat([gt_kpsA_original, est_kpsB]).cpu()
    out_path = f'{out}/transfers'
    ray.init()
    out = batch_overlay(imgs, kps, None, out_path, unique_color=True, size=10)
    grid = images2grid(out, nrow=num_to_vis, normalize=True, range=(0, 255))
    grid_path = f'{out_path}/transfer_grid.png'
    Image.fromarray(grid).save(grid_path)
    # Also save the congealed images:
    congealed = t(torch.cat([imgsA, imgsB]), output_resolution=imgsB.size(-1))
    congealed_path = f'{out_path}/congealed.png'
    save_image(congealed, congealed_path, nrow=num_to_vis, normalize=True, range=(-1, 1))
    print(f'Saved visualization to {grid_path} and {congealed_path}')


@torch.inference_mode()
def pck_transfer(t, loader, alpha=0.1, num_pairs=10000, device='cuda', quiet=True, transfer_both_ways=True,
                 permutation=None, match_flows=True, **stn_forward_kwargs):
    """
    Computes the PCK-Transfer evaluation metric.
    :param t: Spatial Transformer network
    :param loader: DataLoader for PCK pairs
    :param alpha: (float or list of floats) Thresholds at which to compute PCK-Transfer
    :param num_pairs: (int) The number of pairs to evaluate
    :param device: Device to evaluate on
    :param quiet: If False, displays progress bar
    :param transfer_both_ways: If True, evaluates both A --> B and B --> A transfers
    :param permutation: list of ints or 1D int tensor which indicates how key points should be permuted when an image mirrors
    :param match_flows: If True, infer mirror operations on inputs
    :param stn_forward_kwargs: Any arguments for SpatialTransformer.forward
    :return: A tensor of size-alpha representing the PCK-Transfer for each alpha threshold
    """
    pairs_per_gpu = num_pairs // get_world_size()
    excess_pairs = num_pairs % get_world_size()
    pairs_needed_on_this_gpu = pairs_per_gpu + (get_rank() < excess_pairs)  # some GPUs may process 1 extra pair
    pairs_seen = 0
    key_points_seen = 0
    num_alphas = len(alpha) if isinstance(alpha, (list, tuple)) else 1
    correct = torch.zeros(num_alphas, device=device)
    alpha = torch.tensor(alpha, device=device).view(1, num_alphas)
    pbar = None if quiet or not primary() else tqdm(total=pairs_needed_on_this_gpu)
    while pairs_seen < pairs_needed_on_this_gpu:
        # Load image and key point pairs:
        d = next(loader)
        batch_size = d['imgsA'].size(0)
        pairs_still_needed = pairs_needed_on_this_gpu - pairs_seen
        if batch_size > pairs_still_needed:  # Make sure not to overshoot the number of pairs evaluated:
            d = {key: val[:pairs_still_needed] for key, val in d.items()}
        imgsA, imgsB, gt_kpsA, gt_kpsB = d['imgsA'].to(device), d['imgsB'].to(device), \
                                         d['kpsA'].to(device), d['kpsB'].to(device)
        if gt_kpsA.size(-1) == 3:  # (x, y, visibility):
            visible_kps = gt_kpsA[..., 2:3] * gt_kpsB[..., 2:3]  # Create a mask to ignore non-visible key points
            gt_kpsA, gt_kpsB = gt_kpsA[..., :2].clone(), gt_kpsB[..., :2].clone()  # Remove visibility information
        else:  # Assume all key points are visible:
            visible_kps = torch.ones(gt_kpsA.size(0), gt_kpsA.size(1), 1, device=device)
        if match_flows:
            imgsA, imgsB, gt_kpsA, gt_kpsB, _ = t.match_flows(imgsA, imgsB, gt_kpsA, gt_kpsB, permutation, **stn_forward_kwargs)
        # Transfer the key points from imgsA to imgsB:
        est_kpsB = t.transfer_points(imgsA, imgsB, gt_kpsA, **stn_forward_kwargs)
        # Determine which key points were accurately mapped:
        if 'threshB' not in d:  # alpha threshold (used for CUB)
            imgB_thresh = torch.tensor(max(imgsB.size(-2), imgsB.size(-1)), device=device)
        else:  # alpha_bbox threshold (used for SPair-71K categories)
            imgB_thresh = (d['scaleB'] * d['threshB']).to(device)
        thresholdB = alpha * imgB_thresh.view(-1, 1)  # (batch_size, num_alphas)
        # Compute accuracies at each specified alpha threshold. correct_batch is shape (N, num_kps, num_alphas):
        err_A2B = (est_kpsB - gt_kpsB).norm(dim=-1).unsqueeze_(-1)
        correct_batch_A2B = err_A2B <= thresholdB.unsqueeze_(1)
        correct += correct_batch_A2B.mul(visible_kps).sum(dim=(0, 1))
        if transfer_both_ways:
            est_kpsA = t.transfer_points(imgsB, imgsA, gt_kpsB, **stn_forward_kwargs)
            if 'threshA' not in d:  # alpha threshold (used for CUB)
                imgA_thresh = torch.tensor(max(imgsA.size(-2), imgsA.size(-1)), device=device)
            else:  # alpha_bbox threshold
                imgA_thresh = (d['scaleA'] * d['threshA']).to(device)
            thresholdA = alpha * imgA_thresh.view(-1, 1)  # (batch_size, num_alphas)
            err_B2A = (est_kpsA - gt_kpsA).norm(dim=-1).unsqueeze_(-1)
            correct_batch_B2A = err_B2A <= thresholdA.unsqueeze_(1)
            correct += correct_batch_B2A.mul(visible_kps).sum(dim=(0, 1))
        pairs_seen += imgsA.size(0)
        key_points_seen += visible_kps.sum() * (1 + transfer_both_ways)
        if pbar is not None:
            pbar.update(imgsA.size(0))
    assert pairs_seen == pairs_needed_on_this_gpu
    # Normalize by the number of pairs observed times the number of key points per-image:
    total_visible_kps_seen = all_gather(key_points_seen.view(1)).sum()
    pck_alpha = all_gather(correct, cat=False).sum(dim=0).float() / total_visible_kps_seen
    return pck_alpha


if __name__ == '__main__':
    parser = base_eval_argparse()
    # PCK-Transfer hyperparameters:
    parser.add_argument("--alphas", default=[0.1, 0.05, 0.01], type=float, nargs='+', help='Thresholds at which to optionally monitor PCK')
    parser.add_argument("--num_pck_pairs", default=None, type=int, help='Number of pairs to evaluate (None=infer)')
    parser.add_argument("--transfer_both_ways", action='store_true', help='If specified, evaluates A --> B transfers '
                                                                          'as well as B --> A')
    parser.add_argument("--vis_transfer", action='store_true', help='If specified, saves a png visualizing key point '
                                                                    'transfers')
    parser.add_argument("--num_bootstrap", default=0, type=int, help='If greater than zero, also run bootstrapping to '
                                                                     'estimate standard deviations. We use 100 in the '
                                                                     'paper where we report error bars')
    parser.add_argument("--out", default='visuals', type=str, help='Directory to save visualizations')
    args = parser.parse_args()
    args.distributed = setup_distributed(args.local_rank)
    t_ema = load_stn(args)
    run_pck_transfer(args, t_ema)
    if args.num_bootstrap > 0:  # bootstrap standard deviations
        run_pck_bootstrap(args, t_ema)
