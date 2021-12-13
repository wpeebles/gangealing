"""
This script directly applies our method to video, finding dense correspondences across time in an input video. This works
by applying GANgealing per-frame without using any temporal information.
"""
import torch
import numpy as np
import math
from datasets import img_dataloader
from models import SpatialTransformer
from utils.vis_tools.helpers import images2grid, save_video, save_image, load_dense_label, load_cluster_dense_labels, load_pil, splat_points, get_plotly_colors, get_colorscale
from utils.distributed import setup_distributed, primary, all_gather
from applications import base_eval_argparse, load_stn, determine_flips
from tqdm import tqdm
from glob import glob
import os


def grid2vid(list_of_grids):
    # Takes a list of (H, W, C) images (or image grids), runs an all_gather to collect across GPUs and then
    # prepares the images to be saved as a video by the save_video function.
    frames = torch.tensor(np.stack(list_of_grids), device='cuda')
    frames = gather_and_permute(frames)
    frames = [frame for frame in frames.cpu().numpy()]
    return frames


def gather_and_permute(x):
    # This function does all_gather but takes into account the stride of the data created by the distributed
    # data sampler in PyTorch.
    x = all_gather(x, cat=False).transpose(1, 0)
    x = x.reshape(-1, *x.size()[2:])
    return x


def create_output_folder(args, clustering=False):
    video_path = f'{args.out}/video_{os.path.basename(os.path.normpath(args.real_data_path))}'
    if clustering:
        if isinstance(args.cluster, list):  # visualize multiple clusters simultaneously
            video_path = f'{video_path}_{"".join([str(ix) for ix in args.cluster])}'
        elif isinstance(args.cluster, int):  # visualize just one cluster
            video_path = f'{video_path}_{args.cluster}'
    if primary() and args.save_frames:
        os.makedirs(f'{video_path}/frames', exist_ok=True)
        os.makedirs(f'{video_path}/congealing_frames', exist_ok=True)
    elif primary():
        os.makedirs(f'{video_path}', exist_ok=True)
    return video_path


def create_average_image_vis(args, points_per_cluster, video_path, nrow):
    labeled_average_images = []
    for cluster in range(args.num_heads):
        # expects average images are named, e.g., "cat_average_cluster0.png", "cat_average_cluster1.png", etc.
        args.average_path = args.average_path.replace(f'cluster{max(cluster - 1, 0)}', f'cluster{cluster}')
        average_image = load_pil(args.average_path, resolution=args.resolution)
        labeled_average_image = splat_points(average_image, points_per_cluster[cluster].float(), sigma=args.sigma,
                                             opacity=args.opacity, colorscale=get_colorscale(cluster))
        labeled_average_images.append(labeled_average_image)
    labeled_average_images = torch.cat(labeled_average_images, 0)  # (K, C, H, W)
    save_image(labeled_average_images, f'{video_path}/labeled_averages.png', normalize=True, range=(-1, 1), nrow=nrow)
    return labeled_average_images


def number_of_clusters_annotated(path):
    # This function checks how many clusters have average congealed images saved to disk.
    path = path.rstrip('/')
    filename, extension = os.path.splitext(path)
    if not filename.endswith(f'cluster0'):
        num_annos = 1
    else:
        num_annos = len(glob(f'{filename[:-1]}*{extension}'))
    return num_annos


@torch.inference_mode()
def run_gangealing_on_video(args, t, classifier):
    # Step (0): Set some visualization hyperparameters and create results directory:
    alpha = 0.2
    clustering = args.clustering
    video_path = create_output_folder(args, clustering)
    # Construct dataloader:
    loader = img_dataloader(args.real_data_path, resolution=args.real_size, shuffle=False, batch_size=args.batch,
                            distributed=args.distributed, infinite=False, drop_last=False, return_indices=True)
    num_total = len(loader.dataset)
    num_clusters = args.num_heads if clustering else 1
    nrow = int(math.sqrt(num_clusters))

    # Step (1): Load the points (and optionally colors) that we want to propagate to the input video:
    if clustering:
        points_per_cluster, colors_per_cluster, alpha_channels_per_cluster = \
            load_cluster_dense_labels(args.label_path, args.num_heads, args.resolution, args.objects)
        if args.average_path is not None:  # Optionally create a visualization of all the clusters' dense labels:
            labeled_average_images = create_average_image_vis(args, points_per_cluster, video_path, nrow)
            labeled_average_images = labeled_average_images.unsqueeze(0)  # (1, K, C, H, W)
            inactive_averages = labeled_average_images * alpha - (1 - alpha)  # This can be used later to visualize cluster selection
            C, H, W = labeled_average_images.size()[2:]
        points_per_cluster = [SpatialTransformer.normalize(points, args.real_size, args.resolution) for points in points_per_cluster]
    else:  # unimodal GANgealing:
        points, colors, alpha_channels = load_dense_label(args.label_path, args.resolution, args.objects)
        points = SpatialTransformer.normalize(points, args.real_size, args.resolution)

    # Step (2): Pre-process the RGB colors and alpha-channel values that we want to propagate to the input video:
    if clustering and args.cluster is not None:
        mode = 'fixed_cluster'  # clustering, always propagate from the specified cluster(s)
        if not args.objects:
            colors_per_cluster = [get_plotly_colors(points_per_cluster[cluster].size(1), get_colorscale(cluster)) for cluster in range(args.num_heads)]
        colors = [colors_per_cluster[cluster] for cluster in args.cluster]
        colors = torch.cat(colors, 1)
        alpha_channels = [alpha_channels_per_cluster[cluster] for cluster in args.cluster]
        alpha_channels = torch.cat(alpha_channels, 1)
    elif clustering:
        mode = 'predict_cluster'  # clustering, but only propagate based on the current predicted cluster
        if not args.objects:
            colors = colors_per_cluster = [get_plotly_colors(points.size(1), get_colorscale(cluster)) for cluster, points in enumerate(points_per_cluster)]
        alpha_channels = alpha_channels_per_cluster
    else:
        mode = 'unimodal'  # no clustering (num_heads == 1)
        if not args.objects:
            colors = get_plotly_colors(points.size(1), get_colorscale(None))

    # Step (3): Prepare some variables if we want to display the label we're propagating over the congealed video
    if args.overlay_congealed:
        if clustering:
            congealed_points = [SpatialTransformer.unnormalize(points, args.real_size, args.real_size) for points in points_per_cluster]
            congealed_colors = colors_per_cluster
            congealed_alpha_channels = alpha_channels_per_cluster
        else:
            congealed_points = [SpatialTransformer.unnormalize(points, args.real_size, args.real_size)]
            congealed_colors = [colors]
            congealed_alpha_channels = [alpha_channels]

    # Step (4): Start processing the input video.
    # video_frames will be a list of (N, C, H, W) frames: the augmented reality video with objects/points displayed
    # congealing_frames will be a list of (N, C, H, W) frames: the congealed video (i.e., STN(video))
    # [clustering only] average_frames will be a list of (N, C, H, W) frames: a video that shows which cluster(s) is/are active at each frame
    video_frames, congealing_frames, average_frames = [], [], []
    pbar = tqdm(loader) if primary() else loader
    for (batch, batch_indices) in pbar:
        N = batch.size(0)
        batch = batch.to('cuda')
        # Step (4.1) Propagate correspondences to the next batch of video frames:
        if mode == 'unimodal' or mode == 'predict_cluster':
            batch_flipped, flip_indices, warp_policy, active_cluster_ix = \
                determine_flips(args, t, classifier, batch, cluster=args.cluster, return_cluster_assignments=True)
            if clustering:
                points_in = points_per_cluster[active_cluster_ix.item()]
            else:  # mode == 'unimodal'
                points_in = points.repeat(N, 1, 1)
                # Perform the actual propagation:
            propagated_points = t.uncongeal_points(batch_flipped, points_in, normalize_input_points=False,  # already normalized above
                                                   warp_policy=warp_policy,
                                                   padding_mode=args.padding_mode, iters=args.iters)
            # Flip points where necessary:
            propagated_points[:, :, 0] = torch.where(flip_indices.view(-1, 1),
                                                     args.real_size - 1 - propagated_points[:, :, 0],
                                                     propagated_points[:, :, 0])
        else:  # mode == 'fixed_cluster'
            # Here we need to iterate over every cluster we want to visualize so we can propagate points
            # from each individual cluster to the video frame(s):
            propagated_points, active_cluster_ix = [], []
            for cluster in args.cluster:
                batch_flipped, flip_indices, warp_policy, active_cluster_c = \
                    determine_flips(args, t, classifier, batch, cluster=cluster, return_cluster_assignments=True)
                # Perform the actual propagation:
                points_in_c = points_per_cluster[cluster].repeat(N, 1, 1)
                propagated_points_c = t.uncongeal_points(batch_flipped, points_in_c,
                                                         normalize_input_points=False,  # already normalized above
                                                         warp_policy=warp_policy, padding_mode=args.padding_mode,
                                                         iters=args.iters)
                # Flip points where necessary:
                propagated_points_c[:, :, 0] = torch.where(flip_indices.view(-1, 1), args.real_size - 1 - propagated_points_c[:, :, 0],
                                                           propagated_points_c[:, :, 0])
                propagated_points.append(propagated_points_c)
                active_cluster_ix.append(active_cluster_c)
            propagated_points = torch.cat(propagated_points, 1)
            active_cluster_ix = torch.cat(active_cluster_ix, 0)

        # Select the colorscale for visualization:
        if mode == 'unimodal' or mode == 'fixed_cluster':
            colors_in = colors.repeat(N, 1, 1)
            alpha_channels_in = alpha_channels.repeat(N, 1, 1)
        else:  # predict_cluster code path assumes batch size is 1
            assert active_cluster_ix.size(0) == 1
            colors_in = colors[active_cluster_ix.item()]
            alpha_channels_in = alpha_channels[active_cluster_ix.item()]
        video_frame = splat_points(batch, propagated_points, sigma=args.sigma, opacity=args.opacity, colors=colors_in,
                                   alpha_channel=alpha_channels_in)
        if args.save_frames:
            for frame, index in zip(video_frame, batch_indices):
                fn = f'{video_path}/frames/{index.item()}.png'
                save_image(frame, fn, normalize=True, range=(-1, 1), padding=0)
        else:
            video_frames.append(video_frame)

        # Step (4.2) Visualize the congealed video (STN(video)):
        if clustering:
            batch_flipped, warp_policy = classifier.run_flip_cartesian(batch)
        congealed = t(batch_flipped, output_resolution=args.real_size, warp_policy=warp_policy, unfold=clustering,
                      padding_mode=args.padding_mode, iters=args.iters)  # (N, K, C, H, W) or (N, C, H, W)
        if not clustering:
            congealed = congealed.unsqueeze(1)  # (N, 1, C, H, W)
        if args.overlay_congealed:  # Optionally overlay points on the congealed video frames:
            for cluster in range(num_clusters):
                congealed[:, cluster] = splat_points(congealed[:, cluster], congealed_points[cluster].repeat(N, 1, 1),
                                                     sigma=args.sigma, opacity=args.opacity,
                                                     colors=congealed_colors[cluster].repeat(N, 1, 1),
                                                     alpha_channel=congealed_alpha_channels[cluster].repeat(N, 1, 1))

        # This inactive_clusters stuff below is only relevant for clustering models (highlights the currently active cluster(s)):
        inactive_clusters = congealed * alpha - (1 - alpha)  # -1 corresponds to black
        active_cluster_ix = active_cluster_ix.clamp(max=num_clusters - 1)
        one_hot_cluster = torch.eye(num_clusters, device='cuda')[active_cluster_ix].view(N, -1, num_clusters, 1, 1, 1).transpose(1, 0).sum(dim=0).bool()
        congealed = torch.where(one_hot_cluster, congealed, inactive_clusters)
        if args.save_frames:
            for frame, index in zip(congealed, batch_indices):
                fn = f'{video_path}/congealing_frames/{index.item()}.png'
                save_image(frame, fn, normalize=True, range=(-1, 1), pad_value=-1, nrow=nrow)
        else:
            congealed = [
                images2grid(congealed_i, normalize=True, range=(-1, 1), pad_value=-1, nrow=nrow) for congealed_i in
                congealed]
            congealing_frames.extend(congealed)

        # Step (4.3) For clustering models, show which cluster(s) is/are currently active:
        if clustering and args.average_path is not None:
            current_cluster_average = torch.where(one_hot_cluster, labeled_average_images.expand(N, args.num_heads, C, H, W),
                                                  inactive_averages.expand(N, args.num_heads, C, H, W))
            average = [
                images2grid(average_i, normalize=True, range=(-1, 1), pad_value=-1,
                            nrow=nrow) for average_i in current_cluster_average]
            average_frames.extend(average)
    # Step (5): Save the final mp4 videos:
    if primary() and args.save_frames:  # Load saved frames from disk:
        video_frames = [f'{video_path}/frames/{i}.png' for i in range(num_total)]
        congealing_frames = [f'{video_path}/congealing_frames/{i}.png' for i in range(num_total)]
        save_video(video_frames, args.fps, f'{video_path}/propagated.mp4', filenames=True)
        save_video(congealing_frames, args.fps, f'{video_path}/congealed.mp4', filenames=True)
    elif not args.save_frames:
        video_frames = gather_and_permute(torch.cat(video_frames, 0))[:num_total]
        if primary():
            save_video(video_frames, args.fps, f'{video_path}/propagated.mp4', input_is_tensor=True)
        congealed_frames = grid2vid(congealing_frames)[:num_total]
        if primary():
            save_video(congealed_frames, args.fps, f'{video_path}/congealed.mp4', input_is_tensor=False)
    if len(average_frames) > 0:
        average_frames = grid2vid(average_frames)[:num_total]
        if primary():
            save_video(average_frames, args.fps, f'{video_path}/average.mp4')
    if primary():
        print('Done.')


if __name__ == '__main__':
    parser = base_eval_argparse()
    # Visualization hyperparameters:
    parser.add_argument("--cluster", default=None, type=int, nargs='+',
                        help='If using a clustering model, OPTIONALLY select the cluster(s) to create visualizations '
                             'for. If more than one is specified, tracks will be created for all specified clusters. '
                             'If you leave this as None, the cluster will be predicted dynamically for each frame.')
    parser.add_argument("--label_path", type=str, help='Path to a dense label in congealed space, formatted as '
                                                       'an RGBA image', required=True)
    parser.add_argument("--average_path", type=str, default=None, help='Path to an average image for clustering models. '
                                                                       'This should end in "cluster0.png" if specified.')
    parser.add_argument("--save_frames", action='store_true', help='If specified, saves individual frames to disk as pngs '
                                                                   'in addition to making an mp4. This takes much less '
                                                                   'GPU memory but is slower.')
    parser.add_argument("--resolution", type=int, default=128, help='Resolution at which to load label_path. Making this '
                                                                    'larger will propagate more pixels (i.e., find '
                                                                    'denser correspondences)')
    parser.add_argument("--fps", type=int, default=60, help='FPS of saved videos')
    parser.add_argument("--overlay_congealed", action='store_true', help='If specified, overlays the input dense label '
                                                                         'on the congealed mp4 video')
    parser.add_argument("--objects", action='store_true', help='If specified, loads RGB values from the label '
                                                               '(object propagation). Otherwise, an RGB colorscale will '
                                                               'be created.')
    parser.add_argument("--sigma", type=float, default=1.2, help='Size of the propagated points overlaid on the video')
    parser.add_argument("--opacity", type=float, default=0.7, help='Opacity of the propagated points overlaid on the video')
    parser.add_argument("--out", type=str, default='visuals', help='directory where created videos will be saved')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    args.distributed = setup_distributed(args.local_rank)
    # The classifier is optional and only used with clustering models:
    t_ema, classifier = load_stn(args, load_classifier=True)
    if args.num_heads == 1:
        args.clustering = False
    else:  # Only applies to clustering models:
        if args.average_path is not None:
            assert 'cluster0' in args.average_path, 'if supplying an average_image for clustering models, only select ' \
                                                    'the path ending in "cluster0". The other average images will be ' \
                                                    'automatically loaded.'
            assert number_of_clusters_annotated(args.average_path) == args.num_heads
        if number_of_clusters_annotated(args.label_path) == 1:
            # This is a special code path that allows you to do augmented reality from just a single cluster.
            # Usually, the clustering models require that all clusters have annotated average congealed
            # images, but this path requires only a single cluster to be annotated.
            args.clustering = False
            assert args.average_path is None
            assert args.cluster is not None and len(args.cluster) == 1
            args.cluster = args.cluster[0]
        else:  # The usual clustering code path:
            args.clustering = True
            args.batch = 1  # Different clusters may propagate different numbers of points so batch size has to be 1
    run_gangealing_on_video(args, t_ema, classifier)
