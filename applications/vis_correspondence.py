"""
This script creates videos that showcase the congealing and label propagation learned by a pre-trained GANgealing
model.

Example dataset indices used for visualizations:
Cats, CelebA: 5 9 12 13 15 18 19 20 21
Cars: 9 23 28 37 39 49 53 56 101
Horses: 12 15 21 31 41 81 96 124 132
CUB: 1032 230 603 401 555 95 71 392 621
CUB failure cases: 580 581 582 596 597 599 600 601 607
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np
import math
from datasets import MultiResolutionDataset, img_dataloader
from utils.vis_tools.helpers import images2grid, save_video, save_image, normalize, load_dense_label, splat_points, get_colorscale
from utils.distributed import setup_distributed, primary, get_world_size, synchronize, all_gather, all_gatherv
from models.spatial_transformers.antialiased_sampling import MipmapWarp
from models.spatial_transformers.spatial_transformer import SpatialTransformer, ComposedSTN, unravel_index
from applications import base_eval_argparse, load_stn, determine_flips
from applications.flow_scores import filter_dataset
from tqdm import tqdm
import os


@torch.inference_mode()
def sample_images_and_points(args, t, classifier, device):
    dset = MultiResolutionDataset(args.real_data_path, resolution=args.real_size)
    if args.flow_scores is not None:
        dset = filter_dataset(dset, args.flow_scores, args.fraction_retained)
    data = torch.stack([dset[i] for i in args.dset_indices], 0).to(device)
    # Determine which images need to be flipped:
    data_flipped, flip_indices, warp_policy = determine_flips(args, t, classifier, data, cluster=args.cluster)
    if args.label_path is not None:
        points, colors, alpha_channel = load_dense_label(args.label_path, resolution=args.resolution, load_colors=args.objects)
        points = points.repeat(data.size(0), 1, 1)
        points_normalized = SpatialTransformer.normalize(points, args.output_resolution, args.resolution)
        if args.resolution != args.output_resolution:
            points = SpatialTransformer.convert(points, args.resolution, args.output_resolution).round().long()
    else:
        points = points_normalized = colors = alpha_channel = None
    return data, data_flipped, flip_indices, warp_policy, points, points_normalized, colors, alpha_channel


def pad_grid(grid):
    # Pads the border of the grid on each side by 1. The specific way this
    # padding is done is by linearly extrapolating the flow based on the borders.
    # (N, H, W, 2) --> (N, H+2, W+2, 2)
    grid = grid.permute(0, 3, 1, 2)  # (N, 2, H, W)
    grid = F.pad(grid, (1, 1, 1, 1), mode='replicate')  # (N, 2, H+2, W+2)
    grid = grid.permute(0, 2, 3, 1)  # (N, H+2, W+2, 2)
    # Approximate how the flow might look beyond the grid boundaries via finite differences:
    right = 2 * grid[:, :, -2] - grid[:, :, -3]
    left = 2 * grid[:, :, 1] - grid[:, :, 2]
    bottom = 2 * grid[:, -2] - grid[:, -3]
    top = 2 * grid[:, 1] - grid[:, 2]
    # Replace the border padding with the finite differences padding:
    grid[:, 0] = top
    grid[:, -1] = bottom
    grid[:, :, 0] = left
    grid[:, :, -1] = right
    return grid


@torch.inference_mode()
def nearest_neighbor_within_patch(grid, points, patch_centers, patch_size):
    # For each input point, this function returns the spatial indices of a corresponding point in grid that
    # is most similar to the input point (in the L2 sense). This function differs from nearest_neighbor_global in that
    # the search space is restricted to a patch_size-by-patch_size window around patch_centers. This is substantially
    # faster than nearest_neighbor_global, but relies on the assumption that the nearest neighbor will lie within the
    # window.
    # grid: (N, H, W, 2)
    # points: (N, num_points, 2)
    # patch_centers: (N, num_points, 2)
    # patch_size: int
    N = grid.size(0)  # batch_size
    P = points.size(1)  # num_points
    unfold = nn.Unfold(patch_size, padding=patch_size // 2)
    grid = pad_grid(grid)  # (N, H+2, W+2, 2)
    padded_patch_centers = patch_centers + 1
    flat_centers = (padded_patch_centers[..., 0] + grid.size(1) * padded_patch_centers[..., 1])  # (N, P)
    patches = unfold(grid.permute(0, 3, 1, 2))  # (N, 2*patch_size, num_patches)
    patches = patches.gather(dim=2, index=flat_centers.view(N, 1, P).repeat(1, patches.size(1), 1))  # (N, 2*patch_size^2, P)
    # (N, patch_size, patch_size, P, 1, 2)
    patches_reshaped = patches.reshape(N, 2, patch_size, patch_size, P).permute(0, 2, 3, 4, 1).reshape(N, patch_size, patch_size, P, 1, 2)
    points = points.reshape(N, 1, 1, P, 2, 1)  # (N, 1, 1, P, 2, 1)
    similarities = (patches_reshaped @ points)[..., 0, 0]  # (N, patch_size, patch_size, P)
    # Compute distances as: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    distances = points.pow(2).squeeze(-1).sum(dim=-1) + patches_reshaped.pow(2).sum(dim=-1).squeeze(
        -1) - 2 * similarities
    nearest_neighbors = distances.reshape(N, patch_size * patch_size, P).argmin(dim=1)  # (N, P)
    unflattened = unravel_index(nearest_neighbors, (patch_size, patch_size))  # (N, P, 2)
    # Map from patch-based coordinate system back to global image coordinate system:
    diff = unflattened - torch.tensor([patch_size // 2, patch_size // 2], device='cuda', dtype=torch.long).view(1, 1, 2)
    offset = diff[..., 0] + grid.size(1) * diff[..., 1]
    out_coords = flat_centers + offset
    propagated_coords = unravel_index(out_coords, (grid.size(1), grid.size(2)))  # (N, P, 2)
    # remove the padding:
    propagated_coords_unpadded = propagated_coords - 1
    return propagated_coords_unpadded


@torch.inference_mode()
def visualize_correspondence(args, congealing_frames, propagation_frames):
    # This function simply combines the smoothly_congeal and smoothly_propagate videos into a single
    # mp4 in order to clearly illustrate the corerspondences learned by GANgealing.
    pause_steps = 60  # how many frames to wait after congealing and before propagation
    interp_steps = 60  # how many frames over which to superimpose the mask on congealed images
    end_pause_steps = 5  # number of repeated final frames at the very end of the video
    fully_congealed_frame = torch.from_numpy(congealing_frames[-1]).unsqueeze(0).float()
    labeled_congealed_frame = torch.from_numpy(propagation_frames[0]).unsqueeze(0).float()
    alpha = torch.linspace(0, 1, steps=interp_steps).view(interp_steps, 1, 1, 1)
    interp_frames = fully_congealed_frame.lerp(labeled_congealed_frame, alpha)
    interp_frames = [frame for frame in interp_frames.clamp(0, 255).round().numpy().astype(np.uint8)]
    full_video = congealing_frames + [congealing_frames[-1]] * pause_steps + interp_frames + propagation_frames + [propagation_frames[-1]] * end_pause_steps
    save_video(full_video, fps=args.fps, out_path=f'{args.out}/smooth_correspondence.mp4')


def visualize_label_propagation(args, images, propagated_points, colors, alpha_channels, images_per_frame, initial_propagation_frames=[], save=True):
    num_total_images = images.size(0) * images.size(1)
    assert num_total_images % images_per_frame == 0
    colorscale = get_colorscale(args.cluster)
    images = images.view(-1, 3, args.output_resolution, args.output_resolution)
    propagated_points = propagated_points.view(-1, propagated_points.size(2), 2)
    if args.objects:
        colors = colors.repeat(propagated_points.size(0), 1, 1) if colors is not None else None
    alpha_channels = alpha_channels.repeat(propagated_points.size(0), 1, 1)
    propagated_frames = []
    for i in range(0, images.size(0), args.splat_batch):
        propagated_frames.append(splat_points(images[i:i+args.splat_batch], propagated_points[i:i+args.splat_batch],
                                              sigma=args.sigma, opacity=args.opacity,
                                              colorscale=colorscale, colors=colors[i:i+args.splat_batch] if colors is not None else None,
                                              alpha_channel=alpha_channels[i:i+args.splat_batch]).cpu())
    propagated_frames = torch.cat(propagated_frames, 0)
    propagated_frames = propagated_frames.view(-1, images_per_frame, 3, args.output_resolution, args.output_resolution)
    frames = initial_propagation_frames
    nrow = int(math.sqrt(images_per_frame))
    for frame in propagated_frames:
        frame = images2grid(frame, nrow=nrow, normalize=True, range=(-1, 1))
        frames.append(frame)
    frames = frames[::-1]  # Reverse the video
    if save:
        save_video(frames, args.fps, f'{args.out}/smoothly_propagate.mp4', filenames=False)
    return frames


def make_flip_frames(data, flipping_grid, identity_grid, warper, length, nrow):
    congealed_frames, _, _, _ = smoothly_sample_image(flipping_grid, identity_grid, warper, data, length, nrow)
    return [frame for frame in congealed_frames]


def flip_grid(grid, flip_indices):
    grid = grid.clone()
    grid[..., 0] = torch.where(flip_indices.view(1, -1, 1, 1), -grid[..., 0], grid[..., 0])
    return grid


def get_patch_size(length):
    # length controls the number of steps over which the sampling grid is interpolated from the identity
    # grid to the predicted congealing grid. When length is small, pixels can move a lot from one timestep to the next,
    # and hence the window over which we search for nearest neighbors needs to increase commensurately. This function
    # is a basic heuristic to make sure patch_size isn't too small.
    patch_size = math.ceil(9 * max(1, 240 / length))  # heuristic formula to increase patch_size as length decreases
    if patch_size % 2 == 0:  # make sure patch_size is odd
        patch_size += 1
    return patch_size


@torch.inference_mode()
def smoothly_sample_image(grid, identity_grid, warper, data, length, nrow, points=None, patch_centers=None):
    out_frame = []
    out_points = []
    out_image = []
    patch_size = get_patch_size(length)
    for frame_ix in tqdm(range(length)):
        # alpha interpolates the regressed flow to the identity flow. Below we smoothly interpolate alpha between
        # 0 and 1 using cosine annealing:
        alpha = 1 - 0.5 * (1 + torch.cos(torch.tensor(math.pi * frame_ix / (length - 1)))).to(data.device)
        grid_t = identity_grid.lerp(grid, alpha.view(1, 1, 1, 1))
        congealed = warper(data, grid_t)
        image_grid = images2grid(congealed, nrow=nrow, normalize=True, range=(-1, 1), scale_each=False)
        out_frame.append(image_grid)
        out_image.append(congealed)
        if points is not None:  # Propagate points according to grid_t:
            propagated_points = nearest_neighbor_within_patch(grid_t, points, patch_centers, patch_size=patch_size)
            patch_centers = propagated_points
            out_points.append(propagated_points.float())
    if len(out_points) > 0:
        out_points = torch.stack(out_points, 0)
    out_image = torch.stack(out_image, 0)
    return out_frame, out_points, out_image, patch_centers


@torch.inference_mode()
def smoothly_congeal_and_propagate(args, t, classifier):
    device = 'cuda'
    colorscale = get_colorscale(args.cluster)
    # Sample real images and determine if they need to be flipped. Also load dense congealing annotations if provided:
    data, data_flipped, flip_indices, warp_policy, congealed_points, normalized_congealed_points, colors, alpha_channels = \
        sample_images_and_points(args, t, classifier, device)
    # Predict the congealing sampling grids for each image:
    _, grids = t(data_flipped, return_intermediates=True, warp_policy=warp_policy, padding_mode=args.padding_mode,
                 iters=args.iters)
    if not args.vis_in_stages:  # Model only the final transformation directly:
        grids = [grids[-1]]
    grids = torch.stack(grids)  # (num_stages, N, H, W, 2)
    # Update the predicted sampling grids so they perform flipping where needed:
    grids = flip_grid(grids, flip_indices.view(1, -1, 1, 1))
    # Resize the sampling grid if needed:
    if args.output_resolution != args.flow_size:
        grids = grids.reshape(-1, args.flow_size, args.flow_size, 2)
        grids = F.interpolate(grids.permute(0, 3, 1, 2), scale_factor=args.output_resolution / args.flow_size,
                             mode='bilinear').permute(0, 2, 3, 1)
        grids = grids.reshape(-1, data.size(0), args.output_resolution, args.output_resolution, 2)
    # Create an appropriately-sized identity sampling grid (N, output_resolution, output_resolution, 2):
    identity_grid = F.affine_grid(torch.eye(2, 3).unsqueeze(0).repeat(data.size(0), 1, 1),
                                  (data.size(0), 3, args.output_resolution, args.output_resolution)).to(device)

    num_stages = grids.size(0)
    # Append the identity grid as grids[0]:
    flipping_grid = flip_grid(identity_grid, flip_indices)
    grids = torch.cat([flipping_grid.unsqueeze(0), grids], 0)  # (num_stages + 1, N, H, W, 2)
    # Create the interpolator:
    warper = MipmapWarp(3.5).to(device)
    nrow = int(math.sqrt(data.size(0)))
    if args.label_path is not None:
        full_grid = grids[-1]
        # These are where the dense annotations of congealed images map to in the original unaligned images:
        normalized_unaligned_space_points = F.grid_sample(full_grid.permute(0, 3, 1, 2),
                                                          normalized_congealed_points.unsqueeze(2).float(),
                                                          padding_mode='border').squeeze(3).permute(0, 2, 1)
        # Same as above, except values are in the range [0, output_resolution] instead of [-1, 1]:
        unaligned_space_points_unclamped = SpatialTransformer.unnormalize(normalized_unaligned_space_points,
                                                                          args.output_resolution, args.output_resolution)
        unaligned_space_points = unaligned_space_points_unclamped.round().long().clamp(0, args.output_resolution - 1)
        patch_centers = unaligned_space_points
        patch_centers[..., 0] = torch.where(flip_indices.view(-1, 1), args.output_resolution - 1 - patch_centers[..., 0],
                                            patch_centers[..., 0])
        congealed_patch_centers = congealed_points
    else:  # We can ignore all of these variables if we aren't propagating annotations:
        normalized_unaligned_space_points = unaligned_space_points = patch_centers = None
    congealed_frames = []
    propagated_points = []
    congealed_images = []
    initial_propagation_frames = []
    if args.stage_flip:
        flip_frames = make_flip_frames(data, flipping_grid, identity_grid, warper, args.flip_length, nrow)
        congealed_frames.extend(flip_frames)
        if args.label_path is not None:
            data_output_res = warper(data, identity_grid)
            # data_output_res = F.interpolate(data, args.output_resolution, mode='bilinear')  # a bit too aliased
            splatted_data = splat_points(data_output_res, unaligned_space_points_unclamped, sigma=args.sigma,
                                         opacity=args.opacity, colorscale=colorscale,
                                         colors=colors.repeat(data_output_res.size(0), 1, 1) if colors is not None else None,
                                         alpha_channel=alpha_channels.repeat(data_output_res.size(0), 1, 1))
            prop_flip_frames = make_flip_frames(splatted_data, flipping_grid, identity_grid, warper, args.flip_length, nrow)
            initial_propagation_frames = prop_flip_frames
    for i in range(num_stages):  # Smoothly interpolate between warps:
        congealed_frames_stage, propagated_points_stage, congealed_images_stage, patch_centers = \
            smoothly_sample_image(grids[i+1], grids[i], warper, data, args.length, nrow,
                                  normalized_unaligned_space_points, patch_centers)
        propagated_points.append(propagated_points_stage)
        congealed_images.append(congealed_images_stage)
        congealed_frames.extend(congealed_frames_stage)
    if args.label_path is not None:  # Run the interpolation in reverse to make the point propagation consistent:
        for i in range(num_stages):
            alpha = torch.linspace(0, 1, steps=args.length, device=device).view(args.length, 1, 1, 1)
            # Run the point propagation in reverse order (from congealed --> unaligned)
            _, propagated_points_stage_rev, _, congealed_patch_centers = \
                smoothly_sample_image(grids[-i - 2], grids[-i - 1], warper, data, args.length, nrow,
                                      normalized_unaligned_space_points, congealed_patch_centers)
            # Interpolate the bidirectional propagation predictions:
            propagated_points[-i - 1].lerp_(propagated_points_stage_rev.flip(0, ), alpha)

        # Save the propagation video:
        congealed_images = torch.cat(congealed_images)
        propagated_points = torch.cat(propagated_points)
        propagation_frames = visualize_label_propagation(args, congealed_images, propagated_points, colors,
                                                         alpha_channels, data.size(0), initial_propagation_frames)
        visualize_correspondence(args, congealed_frames, propagation_frames)
    else:
        propagation_frames = None
    save_video(congealed_frames, fps=60, out_path=f'{args.out}/smoothly_congeal.mp4')
    return congealed_frames, propagation_frames


def divide_real_images_into_clusters(loader, classifier, cluster, num_clusters, min_needed_per_cluster=None, path=None):
    if path is not None and os.path.isfile(path):  # load cached assignments
        cluster2indices = torch.load(path)
        print('loaded assigned cluster indices')
    else:  # compute assignments and optionally cache them
        device = 'cuda'
        cluster2indices = {i: [] for i in range(num_clusters)}
        totals = torch.zeros(num_clusters)
        min_needed_per_cluster = math.ceil(min_needed_per_cluster / get_world_size())
        pbars = [tqdm(total=min_needed_per_cluster) for _ in range(num_clusters)] if primary() else None
        [pbar.set_description(f'cluster {i}') for i, pbar in enumerate(pbars)]
        for (data, dataset_indices) in loader:
            data = data.to(device)
            predictions = classifier.assign(data)
            for p, i in zip(predictions, dataset_indices):
                assignment = p.item() % num_clusters  # The modulo handles flipping
                cluster2indices[assignment].append(i.item())
                totals[assignment] += 1
                if primary():
                    pbars[assignment].update(1)
            done = min_needed_per_cluster is not None and (totals >= min_needed_per_cluster).all().item()
            if done:
                break
        synchronize()
        cluster2indices = {i: all_gatherv(torch.tensor(cluster2indices[i], device=device)).tolist() for i in range(num_clusters)}
        if path is not None and primary():
            torch.save(cluster2indices, path)
            print(f'Saved assigned cluster indices to {path}')
    indices = cluster2indices[cluster]
    return indices


def create_average_image(args, t, classifier, loader, warper, alpha, output_resolution, warp_index=None,
                         identity_grid=None, flip=None, **stn_kwargs):
    device = 'cuda'
    num_images_per_gpu = args.n_mean // get_world_size()
    assert num_images_per_gpu * get_world_size() == args.n_mean
    assert (num_images_per_gpu // args.batch) == (
                num_images_per_gpu / args.batch), 'Batch size needs to evenly divide samples needed'
    average_image = 0
    total = 0
    total_iters = 0
    for data in loader:
        data = data.to(device)
        data_flipped, flip_indices, warp_policy = determine_flips(args, t, classifier, data, cluster=args.cluster)
        if warp_index >= 0:  # congealing
            _, grids = t(data_flipped, warp_policy=warp_policy, return_intermediates=True, **stn_kwargs)
            grid = grids[warp_index]
            grid = flip_grid(grid, flip_indices)
            if warp_index == 0:  # start from identity grid
                base_grid = identity_grid.repeat(data.size(0), 1, 1, 1)
            else:  # start from previous congealing stage
                base_grid = grids[warp_index - 1]
            base_grid = flip_grid(base_grid, flip_indices)
        else:  # only flip
            assert flip and args.stage_flip
            grid = flip_grid(identity_grid.repeat(data.size(0), 1, 1, 1), flip_indices)
            base_grid = identity_grid
        if output_resolution != grid.size(1):
            grid = F.interpolate(grid.permute(0, 3, 1, 2), scale_factor=output_resolution / grid.size(1),
                                 mode='bilinear').permute(0, 2, 3, 1)
        if output_resolution != base_grid.size(1):
            base_grid = F.interpolate(base_grid.permute(0, 3, 1, 2), scale_factor=output_resolution / base_grid.size(1),
                                      mode='bilinear').permute(0, 2, 3, 1)
        grid = base_grid.lerp(grid, alpha)
        congealed = warper(data, grid)
        N = congealed.size(0)  # Assumes H == W
        if (total + N) > num_images_per_gpu:
            N = num_images_per_gpu - total
        average_image += congealed[:N].sum(dim=0, keepdim=True)
        total += N
        total_iters += 1
        if total >= num_images_per_gpu:
            break
    assert total == num_images_per_gpu, f'Needed {num_images_per_gpu} but got {total}'
    synchronize()
    average_image = all_gather(average_image / num_images_per_gpu).mean(dim=0)
    return average_image


@torch.inference_mode()
def average_and_congeal(args, t, classifier):
    device = 'cuda'
    colorscale = get_colorscale(args.cluster)
    num_stages = len(t.stns) if isinstance(t, ComposedSTN) and args.vis_in_stages else 1
    num_stages += args.stage_flip
    clustering = args.num_heads > 1
    loader = img_dataloader(args.real_data_path, resolution=args.real_size, shuffle=False, batch_size=args.batch,
                            distributed=args.distributed, return_indices=clustering, infinite=False)
    if clustering:  # clustering-specific step:
        path = f'visuals/cluster2indices_{os.path.basename(os.path.normpath(args.real_data_path))}.pt'
        indices = divide_real_images_into_clusters(loader, classifier, args.cluster, args.num_heads, args.n_mean, path)
        dataset = MultiResolutionDataset(args.real_data_path, resolution=args.real_size)
        cluster_subset = Subset(dataset, indices)
        loader = img_dataloader(dset=cluster_subset, distributed=args.distributed, shuffle=False, batch_size=args.batch,
                                infinite=False)
    identity_grid = F.affine_grid(torch.eye(2, 3).unsqueeze(0),
                                  (1, 3, args.output_resolution, args.output_resolution)).to(device)
    warper = MipmapWarp(3.5).to(device)
    frames = []
    for i in range(num_stages):  # Iterate over STN stages
        average_images = []
        length = args.length if not args.stage_flip or i > 0 else args.flip_length
        pbar = tqdm(range(length)) if primary() else range(length)
        for frame_ix in pbar:  # Interpolate between warps
            flip = (i == 0) and args.stage_flip
            alpha = 1 - 0.5 * (1 + torch.cos(torch.tensor(math.pi * frame_ix / (length - 1)))).to(device)
            average_image = create_average_image(args, t, classifier, loader, warper, alpha,
                                                 warp_index=i - args.stage_flip, identity_grid=identity_grid,
                                                 flip=flip, iters=args.iters, output_resolution=args.output_resolution,
                                                 padding_mode=args.padding_mode)
            average_images.append(average_image)
            if frame_ix == 0 and i == 0 and primary():
                save_image(average_images[0], f'{args.out}/initial_average.png', normalize=True)
        frames.extend(average_images)
        if primary():
            save_image(average_images[-1], f'{args.out}/stage{i}_average.png', normalize=True)
    if primary():
        frames = torch.stack(frames)
        frames = normalize(frames)
        if args.label_path is not None:  # Save the annotated average congealed image:
            _, _, _, _, congealed_points, _, colors, alpha_channels = sample_images_and_points(args, t, classifier, device)
            last_frame = frames[-1].unsqueeze(0).mul(2).add(-1)
            propagated_average = splat_points(last_frame, congealed_points.float()[0:1], sigma=args.sigma, opacity=args.opacity,
                                              colorscale=colorscale, colors=colors, alpha_channel=alpha_channels)
            save_image(propagated_average, f'{args.out}/labeled_average.png', normalize=True, range=(-1, 1))
            alpha = torch.linspace(0, 1, steps=60, device=device).view(60, 1, 1, 1)
            interp_frames = last_frame.lerp(propagated_average, alpha)
            interp_frames = normalize(interp_frames, amin=-1, amax=1)
            pause_steps = 60
            frames = torch.cat([frames, frames[-1].unsqueeze(0).repeat(pause_steps, 1, 1, 1), interp_frames,
                                interp_frames[-1].unsqueeze(0).repeat(5, 1, 1, 1)], 0)
        frames = frames.mul(255.0).round().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        frames = [frame for frame in frames]
        save_video(frames, args.fps, f'{args.out}/smoothly_average.mp4', codec='libx264')
        save_video(frames, args.fps, f'{args.out}/smoothly_average.avi', codec='png')


if __name__ == '__main__':
    parser = base_eval_argparse()
    # Visualization hyperparameters:
    parser.add_argument("--cluster", default=None, type=int,
                        help='if using a clustering model, select the cluster index to create visualizations for')
    parser.add_argument("--length", type=int, default=240, help='The number of frames to generate. Larger number will '
                                                                'produce smoother visualizations. ')
    parser.add_argument("--flip_length", type=int, default=40, help='The number of frames to generate for the flipping'
                                                                    'transformation. ')
    parser.add_argument("--vis_in_stages", action='store_true', help='If specified, visualize each individual Spatial '
                                                                     'Transfer in sequence as opposed to visualizing '
                                                                     'the final composed warp all at once')
    parser.add_argument("--stage_flip", action='store_true', help='If specified, visualizes a flip before '
                                                                  'the first warp (only used for --vis_in_stages)')
    parser.add_argument("--n_mean", type=int, default=-1, help='The number of images used to create the average image '
                                                               'visualizations. If n_mean=-1, then no average image '
                                                               'visualizations will be created.')
    parser.add_argument("--output_resolution", type=int, default=256,
                        help='Resolution of the output video. Note that the regressed flow will be upsampled to this '
                             'resolution. This produces a very high quality output image (much higher quality than '
                             'upsampling in pixel space directly) as long as output_resolution is at most '
                             'real_size.')
    parser.add_argument("--resolution", type=int, default=256, help='Resolution of the flow field. Making this larger '
                                                                    'will construct denser correspondences')
    parser.add_argument("--dset_indices", type=int, nargs='+', default=list(range(4)),
                        help='Select the images (dataset indices) to create visualizations for')
    parser.add_argument("--flow_scores", default=None, type=str,
                        help='Path to pre-computed flow scores to filter dataset (see flow_scores.py for more info)')
    parser.add_argument("--fraction_retained", default=1.0, type=float,
                        help='Fraction of dataset images to retain based on flow scores')
    parser.add_argument("--label_path", type=str, default=None, help='Path to a dense label in congealed space, '
                                                                     'formatted as an RGBA image')
    parser.add_argument("--fps", type=int, default=60, help='FPS of saved videos')
    parser.add_argument("--objects", action='store_true', help='If specified, loads RGB values from the label '
                                                               '(object propagation)')
    parser.add_argument("--sigma", type=float, default=1.2, help='Size of the propagated points overlaid on the video')
    parser.add_argument("--opacity", type=float, default=0.7, help='Opacity of the propagated points overlaid on the video')
    parser.add_argument("--splat_batch", type=int, default=100, help='Batch size for the splatting operation')
    parser.add_argument("--out", type=str, default='visuals', help='directory where created videos will be saved')
    args = parser.parse_args()
    if args.num_heads > 1:  # Only applies to clustering models:
        assert args.cluster is not None, 'Must add --cluster <index> to select a cluster to visualize'
    os.makedirs(args.out, exist_ok=True)
    create_average_visualization = args.n_mean > 0
    args.distributed = setup_distributed(args.local_rank) if create_average_visualization else False
    # The classifier is optional and only used with clustering models:
    t_ema, classifier = load_stn(args, load_classifier=True)
    if primary():  # This part is fast on a single GPU, no need for distributed:
        smoothly_congeal_and_propagate(args, t_ema, classifier)
    if create_average_visualization:
        average_and_congeal(args, t_ema, classifier)
