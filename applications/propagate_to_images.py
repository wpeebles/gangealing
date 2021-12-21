"""
This script runs GANgealing on images. It computes and saves output aligned images, the average congealed image and
propagated edits. To generate fancy videos showing warping, use vis_correspondence.py instead.

Example success case --dset_indices used for visualizations:

Bicycle: 72 179 225 16 90 48 227 235 249
Cat: 1922 2363 8558 7401 9750 7432 2105 53 1946
Dog: 180 199 15 241 121 124 257 203 208
TV: 234 6 111 19 8 89 1 223 219
CUB: 1629 621 219 1515 1430 603 392 220 1519
In-The-Wild CelebA: 14 739 738 1036 506 534 517 2054 15
CUB failure cases: 580 581 582 596 597 599 600 601 607
"""
import torch
from torch.utils.data import Subset
import math
import os

from applications import base_eval_argparse, load_stn, determine_flips
from applications.vis_correspondence import sample_images_and_points, divide_real_images_into_clusters
from datasets import MultiResolutionDataset, img_dataloader
from utils.vis_tools.helpers import save_image, splat_points, load_pil
from utils.vis_tools.training_vis import run_loader_mean
from utils.distributed import setup_distributed, primary


def write(image_batch, folder_name):
    # Saves image_batch to disk
    nrow = int(math.sqrt(image_batch.size(0)))
    # Save grid of output images:
    save_image(image_batch, f'{args.out}/{folder_name}_grid.png', normalize=True, range=(-1, 1), nrow=nrow, pad_value=-1.0, padding=3)
    if args.save_individual_images:  # Save each output image individually in a new folder:
        os.makedirs(f'{args.out}/{folder_name}', exist_ok=True)
        for i, image in enumerate(image_batch):
            save_image(image.unsqueeze_(0), f'{args.out}/{folder_name}/{i:03}.png', normalize=True, range=(-1, 1), nrow=1, padding=0)


def expand_rank3_batch(tensor, batch_size):
    if tensor is not None:
        return tensor.expand(batch_size, -1, -1)


@torch.inference_mode()
def make_visuals(args, t, classifier):
    # (1) Real images:
    print('Loading and saving real images...')
    reals, reals_flipped, flip_indices, warp_policy, points, points_normalized, colors, alpha_channels = \
        sample_images_and_points(args, t, classifier, device='cuda')
    points_normalized = expand_rank3_batch(points_normalized, reals.size(0))
    colors = expand_rank3_batch(colors, reals.size(0))
    alpha_channels = expand_rank3_batch(alpha_channels, reals.size(0))
    write(reals, 'input_images')
    # (2) Congealed (aligned) images:
    print('Congealing (aligning) images...')
    congealed_reals = t(reals_flipped, padding_mode=args.padding_mode, iters=args.iters,
                        output_resolution=args.output_resolution, warp_policy=warp_policy)
    write(congealed_reals, 'congealed_images')
    # (3) Edit Propagation/ Dense Correspondence:
    if args.label_path is not None:
        print(f'Propagating {args.label_path} to images...')
        # Compute where points_normalized lie in the orginal unaligned images:
        upoints = t.uncongeal_points(reals_flipped, points_normalized, normalize_input_points=False,  # already normalized above
                                     warp_policy=warp_policy, padding_mode=args.padding_mode, iters=args.iters)
        # We need to flip the points wherever our model predicted flips so they are
        # correctly overlaid on the original, unflipped images:
        upoints[:, :, 0] = torch.where(flip_indices.view(-1, 1),
                                       args.real_size - 1 - upoints[:, :, 0],
                                       upoints[:, :, 0])
        propagated_reals = splat_points(reals, upoints, sigma=args.sigma, opacity=args.opacity, colorscale='plasma',
                                        colors=colors, alpha_channel=alpha_channels)
        write(propagated_reals, 'propagated')
        if args.average_path is not None:  # (4) Annotated Average Image:
            average_image = load_pil(args.average_path, args.real_size)
            annotated_average = splat_points(average_image, points.float()[0:1], sigma=args.sigma, opacity=args.opacity,
                                             colorscale='plasma', colors=colors[0:1], alpha_channel=alpha_channels[0:1])
            write(annotated_average, 'average_annotated')
    print(f'All output images can be found at {args.out}')


def average(args, t, classifier):
    # Computes and saves the average congealed (aligned) image.
    print('Computing average image')

    def stn_forward(x, **stn_kwargs):
        data_flipped, flip_indices, warp_policy = determine_flips(args, t, classifier, x, cluster=args.cluster)
        return t(data_flipped, warp_policy=warp_policy, **stn_kwargs)

    loader = img_dataloader(args.real_data_path, resolution=args.real_size, shuffle=False, batch_size=args.batch,
                            distributed=args.distributed, return_indices=args.num_heads > 1, infinite=True)
    if args.num_heads > 1:
        path = f'visuals/cluster2indices_{os.path.basename(os.path.normpath(args.real_data_path))}.pt'
        indices = divide_real_images_into_clusters(loader, classifier, args.cluster, args.num_heads, args.n_mean, path)
        dataset = MultiResolutionDataset(args.real_data_path, resolution=args.real_size)
        cluster_subset = Subset(dataset, indices)
        loader = img_dataloader(dset=cluster_subset, distributed=args.distributed, shuffle=False, batch_size=args.batch,
                                infinite=False)
    _, avg = run_loader_mean(stn_forward, loader, 'cuda', args.n_mean, unfold=False, iters=args.iters,
                             padding_mode=args.padding_mode, output_resolution=args.output_resolution)
    if primary():
        average_path = f'{args.out}/average.png'
        args.average_path = average_path
        save_image(avg, average_path, normalize=True, range=None)
        print(f'Saved average image at {average_path}')


if __name__ == '__main__':
    parser = base_eval_argparse()
    # Visualization hyperparameters:
    parser.add_argument("-s", "--sigma", default=1.3, type=float)
    parser.add_argument("-o", "--opacity", default=0.75, type=float)
    parser.add_argument("--objects", action='store_true', help='If specified, loads RGB values from the label '
                                                               '(object/edit propagation)')
    parser.add_argument("--cluster", default=None, type=int,
                        help='if using a clustering model, select the cluster index to create visualizations for')
    parser.add_argument("--n_mean", type=int, default=-1, help='The number of images used to create the average image '
                                                               'visualizations. If n_mean=-1, then no average image '
                                                               'visualizations will be created.')
    parser.add_argument("--average_path", type=str, default=None, help='Path to an average aligned image. Creates an '
                                                                       'additional visualization showing the label image '
                                                                       'overlaid on the average image. If you\'re using '
                                                                       '--n_mean > 0, this arg be set automatically.')
    parser.add_argument("--output_resolution", type=int, default=None,
                        help='Resolution of the output (congealed) images (default: auto)')
    parser.add_argument("--resolution", type=int, default=256, help='Resolution of the flow field. Making this larger '
                                                                    'will construct denser correspondences')
    parser.add_argument("--dset_indices", type=int, nargs='+', default=None,
                        help='Select the images (dataset indices) to create visualizations for')
    parser.add_argument("--flow_scores", default=None, type=str,
                        help='Path to pre-computed flow scores to filter dataset (see flow_scores.py for more info)')
    parser.add_argument("--fraction_retained", default=1.0, type=float,
                        help='Fraction of dataset images to retain based on flow scores')
    parser.add_argument("--label_path", type=str, default=None, help='Path to a dense label in congealed space, '
                                                                     'formatted as an RGBA image')
    parser.add_argument("--save_individual_images", action='store_true',
                        help='If specified, saves all output images to disk individually '
                             '(default: saves grids of output images)')
    parser.add_argument("--out", type=str, default='visuals', help='directory where created videos will be saved')
    args = parser.parse_args()

    if args.num_heads > 1:  # Only applies to clustering models:
        assert args.cluster is not None, 'Must add --cluster <index> to select a cluster to visualize'
    if args.output_resolution is None:
        args.output_resolution = args.real_size

    os.makedirs(args.out, exist_ok=True)
    create_average_visualization = args.n_mean > 0
    args.distributed = setup_distributed() if create_average_visualization else False
    # The classifier is optional and only used with clustering models:
    t_ema, classifier = load_stn(args, load_classifier=True)
    if create_average_visualization:
        average(args, t_ema, classifier)
    if primary():
        make_visuals(args, t_ema, classifier)
