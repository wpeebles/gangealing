import torch
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from models.losses.loss import sample_gan_supervised_pairs
from models import assign_fake_images_to_clusters
from utils.vis_tools.flow_vis import flow_to_image
from utils.distributed import all_gather, get_world_size, all_reduce, primary
from utils.vis_tools.helpers import images2grid, save_video
from glob import glob
import argparse
import os


@torch.inference_mode()
def run_loader_mean(stn, loader, device, max_eles=12000, unfold=True, **stn_kwargs):
    # Computes the average congealed image over a dataloader of (usually real) images.
    # Also returns the congealed images computed on this process.
    out = []
    total = 0
    for x in loader:
        x = x.to(device)
        out.append(stn(x, unfold=unfold, **stn_kwargs).cpu())
        total += x.size(0)
        if total >= (max_eles // get_world_size()):
            break
    out = torch.cat(out, dim=0)
    means = all_reduce(out, device)
    return out, means


@torch.inference_mode()
def accumulate_means(congealed_images_by_head, device):
    num_heads = len(congealed_images_by_head)
    congealed_images_by_head = [torch.stack(head, 0) for head in congealed_images_by_head]
    # num is of size (num_heads,) and contains the number of images assigned to each cluster:
    num = torch.tensor([head.size(0) for head in congealed_images_by_head], dtype=torch.float, device=device).unsqueeze(0)
    # compute the average image independently per-cluster:
    means = torch.stack([head.sum(dim=0) for head in congealed_images_by_head]).unsqueeze(0).to(device)
    # P is the number of PyTorch processes:
    means = all_gather(means)  # (P, H, C, H, W)
    num = all_gather(num)  # (P, H)
    total_per_head = num.sum(dim=0).view(num_heads, 1, 1, 1)
    means = means.sum(dim=0).div(total_per_head)
    return congealed_images_by_head, means


def pad_heads(congealed_images_by_head, n_sample, num_heads, size):
    # Pad the clusters in case insufficient images were assigned to one (makes visualizations easier)
    for cluster_ix in range(num_heads):
        num_assigned_to_head = len(congealed_images_by_head[cluster_ix])
        if num_assigned_to_head < n_sample:
            diff = n_sample - num_assigned_to_head
            congealed_images_by_head[cluster_ix].extend([torch.zeros(3, size, size)] * diff)
    return congealed_images_by_head


@torch.inference_mode()
def generate_cluster_congeal(stn, generator, ll, loss_fn, resize_fake2stn, z, psi, dim_latent,
                             num_heads, flips, vis_batch_size, n_mean, n_sample, size, device, **stn_kwargs):
    # This function generates fake images, congeals them and assigns them to their respective clusters. The output is
    # a list of length-num_heads, where the i-th element contains the congealed fake images assigned to cluster i.
    # The average images per-cluster are also returned. This function is basically a wrapper around
    # assign_fake_images_to_clusters which organizes the congealed images into their cluster.
    congealed_images_by_head = [[] for _ in range(num_heads)]
    total = 0
    while True:  # process in mini-batches to save memory
        z_in = z[total:total+vis_batch_size]
        # Generate the fake images G(z_in), congeal them and determine which cluster each one belongs to:
        assignments, aligned_pred, _, _, _, _ = \
            assign_fake_images_to_clusters(generator, stn, ll, loss_fn, resize_fake2stn, psi, z_in.size(0), dim_latent,
                                           True, num_heads, flips, device, sample_from_full_res=True, z=z_in, **stn_kwargs)
        CHW = aligned_pred.size()[1:]
        if flips:
            aligned_pred = aligned_pred.reshape(2, z_in.size(0), num_heads, *CHW).permute(1, 0, 2, 3, 4, 5).reshape(
                z_in.size(0), 2 * num_heads, *CHW)
        else:
            aligned_pred = aligned_pred.view(z_in.size(0), num_heads, *CHW)
        # Separate the congealed fake images into their assigned clusters:
        assigned_warps = aligned_pred[torch.arange(z_in.size(0), device=device), assignments.indices]
        for warp, class_ix in zip(assigned_warps, assignments.indices):
            congealed_images_by_head[class_ix.item() % num_heads].append(warp.cpu())  # The modulo handles flipping
        total += z_in.size(0)
        if total >= (n_mean // get_world_size()):
            break
    congealed_images_by_head = pad_heads(congealed_images_by_head, n_sample, num_heads, size)
    congealed_images_by_head, means = accumulate_means(congealed_images_by_head, device)
    return congealed_images_by_head, means


@torch.inference_mode()
def real_cluster_congeal(t_ema, classifier, loader, num_heads, n_mean, n_sample, device, **stn_kwargs):
    congealed_images_by_head = [[] for _ in range(num_heads)]
    total = 0
    for x in loader:
        total += x.size(0)
        x = x.to(device)
        preds = classifier(x)
        classes = preds.argmax(dim=1)  # The modulo handles flipping classes
        flip_indicator = classes >= num_heads  # First num_heads = no flip; second num_heads = flip
        x = torch.where(flip_indicator.reshape(x.size(0), 1, 1, 1), x.flip(3,), x)
        congealed_images = t_ema(x, warp_policy=preds, **stn_kwargs)
        for congealed_image, class_ix in zip(congealed_images, classes):
            congealed_images_by_head[class_ix.item() % num_heads].append(congealed_image.cpu())
        if total >= (n_mean // get_world_size()):
            break
    congealed_images_by_head = pad_heads(congealed_images_by_head, n_sample, num_heads, congealed_images.size(-1))
    congealed_images_by_head, means = accumulate_means(congealed_images_by_head, device)
    return congealed_images_by_head, means


@torch.inference_mode()
def create_fake_visuals(generator, stn, ll, z, resize_fake2stn, psi, n_sample, device, i, writer, **stn_kwargs):
    sample, truncated_sample = sample_gan_supervised_pairs(generator, ll, lambda x: x, psi, n_sample, None,
                                                           True, device, z=z)
    transformed_sample = stn(resize_fake2stn(sample), **stn_kwargs)  # GANgealed GAN samples
    writer.log_image_grid(sample, 'sample', i, n_sample)
    writer.log_image_grid(transformed_sample, 'transformed_sample', i, n_sample)
    writer.log_image_grid(truncated_sample, 'truncated_sample', i, n_sample)


@torch.inference_mode()
def create_training_visuals(generator, t_ema, ll, loader, sample_reals, resize_fake2stn, z, psi,
                            device, n_mean, n_sample, i, writer, **stn_kwargs):
    if loader is not None:  # Create visuals for real images:
        _, mean_transformed_real_imgs = run_loader_mean(t_ema, loader, device, n_mean, **stn_kwargs)
        if primary():
            # Average congealed image:
            writer.log_image_grid(mean_transformed_real_imgs, 'mean_EMA_transformed_real_sample', i, n_sample,
                                  num_heads=1, log_mean_img=False, range=None, scale_each=True)
            # Individually congealed images:
            transformed_real_imgs, real_flow = t_ema(sample_reals, return_flow=True, **stn_kwargs)
            writer.log_image_grid(transformed_real_imgs, 'EMA_transformed_real_sample', i, n_sample, log_mean_img=False)
            if t_ema.is_flow:  # Visualize the flow fields produced by the STN:
                real_flow = flow_to_image(real_flow)
                writer.log_image_grid(real_flow, 'flow_real', i, n_sample, log_mean_img=False, range=(0, 1))

    if primary():  # Create visuals for fake images:
        create_fake_visuals(generator, t_ema, ll, z, resize_fake2stn, psi, n_sample, device, i, writer, **stn_kwargs)


@torch.inference_mode()
def create_training_cluster_visuals(generator, t_ema, ll, loss_fn, loader, resize_fake2stn, z, big_z, psi, device,
                                    n_mean, n_sample, num_heads, flips, vis_batch_size, size, i, writer, **stn_kwargs):
    if loader is not None:  # Create visuals for real images:
        local_transformed_real_imgs, mean_transformed_real_imgs = run_loader_mean(t_ema, loader, device, n_mean, **stn_kwargs)
        if primary():
            # Average (real) congealed images for each cluster. Note that this particular average image
            # does NOT take cluster assignment into account (since we need the cluster classifier to determine
            # that for real images).
            writer.log_image_grid(mean_transformed_real_imgs, 'mean_EMA_transformed_real_sample', i, n_sample,
                                  num_heads=1, log_mean_img=False, range=None, scale_each=True)
            # Visualize several real images congealed to every cluster:
            writer.log_image_grid(local_transformed_real_imgs.view(-1, *local_transformed_real_imgs.size()[2:]),
                    'EMA_transformed_real_sample', i, n_sample, num_heads=num_heads, log_mean_img=False)
            for cluster_ix in range(num_heads):
                # Congealed real images for each cluster. This does NOT take cluster assignment into account.
                writer.log_image_grid(local_transformed_real_imgs[:, cluster_ix], f'EMA_head_{cluster_ix}', i,
                                      n_sample, num_heads=1, log_mean_img=False)

    clustered_fakes, cluster_means = generate_cluster_congeal(t_ema, generator, ll, loss_fn, resize_fake2stn, big_z,
                                                              psi, None, num_heads, flips, vis_batch_size, n_mean,
                                                              n_sample, size, device, **stn_kwargs)
    if primary():  # Create visuals for fake images
        # Average congealed fake images for each cluster. This DOES take cluster assignment into account:
        writer.log_image_grid(cluster_means, 'mean_generated_EMA_transformed_assigned', i, n_sample,
                              num_heads=1, log_mean_img=False, range=None, scale_each=True)
        for cluster_ix in range(num_heads):
            # Congealed fake images for each cluster. This DOES take cluster assignment into account:
            writer.log_image_grid(clustered_fakes[cluster_ix], f'generated_EMA_assigned_head_{cluster_ix}', i,
                                  n_sample, num_heads=1, log_mean_img=False)
        # Other fake image visualizations:
        create_fake_visuals(generator, t_ema, ll, z, resize_fake2stn, psi, n_sample, device, i, writer, **stn_kwargs)


@torch.inference_mode()
def create_training_cluster_classifier_visuals(t_ema, classifier, loader, num_heads, n_mean, n_sample, device, i,
                                               writer, **stn_kwargs):
    local_transformed_assigned, mean_transformed_assigned = real_cluster_congeal(t_ema, classifier, loader, num_heads,
                                                                                 n_mean, n_sample, device, **stn_kwargs)
    if primary():
        # Visualize the average images of each cluster for real images, taking clustering assignment into account.
        writer.log_image_grid(mean_transformed_assigned, 'mean_EMA_transformed_assigned', i, n_sample,
                              num_heads=1, log_mean_img=False, range=None, scale_each=True)
        # Congealed real images for each cluster. This DOES take (predicted) cluster assignment into account.
        for cluster_ix in range(num_heads):
            writer.log_image_grid(local_transformed_assigned[cluster_ix], f'EMA_assigned_head_{cluster_ix}', i,
                                  n_sample, num_heads=1, log_mean_img=False)


class GANgealingWriter(SummaryWriter):

    def __init__(self, results_path, log_images_to_tb=False):
        os.makedirs(os.path.join(results_path, 'checkpoints'), exist_ok=True)
        super().__init__(results_path)
        self.results_path = results_path
        self.log_images_to_tb = log_images_to_tb

    def _log_image_grid(self, images, logging_name, prefix, itr, range=(-1, 1), scale_each=False):
        nrow = max(1, int(images.size(0) ** 0.5))
        ndarr = images2grid(images, return_as_PIL=True, nrow=nrow, padding=2, pad_value=0, normalize=True, range=range,
                            scale_each=scale_each)
        grid = Image.fromarray(ndarr)
        grid.save(f"{self.results_path}/{logging_name}_{str(itr).zfill(7)}.png")
        if self.log_images_to_tb:
            self.add_image(f"{prefix}/{logging_name}", ndarr, itr, dataformats='HWC')

    def log_image_grid(self, images, logging_name, itr, imgs_to_show, log_mean_img=True,
                       mean_range=None, range=(-1, 1), scale_each=False, num_heads=1):
        self._log_image_grid(images[:imgs_to_show], logging_name, "grids", itr, range=range, scale_each=scale_each)
        if log_mean_img:  # Log average images:
            images = images.reshape(images.size(0) // num_heads, num_heads, *images.size()[1:])
            self._log_image_grid(images.mean(dim=0), f'mean_{logging_name}', "means", itr,
                                 range=mean_range, scale_each=True)


def animate_training_visuals():
    MAX_NUM_CLUSTERS = 8  # maximum number of clusters to generate visuals for
    vis_options = ['EMA_transformed_real_sample', 'mean_EMA_transformed_real_sample',
                   'mean_truncated_sample', 'truncated_sample', 'mean_sample', 'sample',
                   'transformed_sample', 'mean_transformed_sample', 'sample_aligned',
                   'EMA_transformed_real_sample_alpha', 'EMA_transformed_assigned', 'mean_EMA_transformed_assigned', 'flow_real',
                   'mean_generated_EMA_transformed_assigned', 'EMA_transformed_identity', 'EMA_correspondence'] + \
                  [f'EMA_head_{i}' for i in range(MAX_NUM_CLUSTERS)] + \
                  [f'EMA_assigned_head_{i}' for i in range(MAX_NUM_CLUSTERS)] + \
                  [f'generated_EMA_assigned_head_{i}' for i in range(MAX_NUM_CLUSTERS)]
    parser = argparse.ArgumentParser(description="Create MP4 videos from frames")

    parser.add_argument("--exps", nargs='+', required=True, help="Names of experiments to generate results for")
    parser.add_argument("--results", type=str, default='results', help="Path to results directory")
    parser.add_argument("--visuals", nargs='+', default=vis_options, help="Which results to visualize")
    parser.add_argument("--fps", type=int, default=60, help="FPS for videos")
    opt = parser.parse_args()

    for exp in opt.exps:
        for visual in opt.visuals:
            opt.vis = visual
            opt.exp = exp
            create_mp4(opt)
    print('Done!')


def create_mp4(opt):
    path = os.path.join(opt.results, opt.exp)
    print(path)
    files = sorted(glob(f'{path}/{opt.vis}_*.png'))
    n_frames = len(files)
    print(f'Found {n_frames} frames')
    if n_frames == 0:
        print('Skipping...')
        return
    out_path = f'visuals/{opt.exp}'
    os.makedirs(out_path, exist_ok=True)
    save_video(list(files), opt.fps, f'{out_path}/{opt.vis}.mp4', filenames=True)


if __name__ == '__main__':
    animate_training_visuals()
