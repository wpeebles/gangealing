import torch


def total_variation_loss(delta_flow, reduce_batch=True):
    # flow should be size (N, H, W, 2)
    reduce_dims = (0, 1, 2, 3) if reduce_batch else (1, 2, 3)
    distance_fn = lambda a: torch.where(a <= 1.0, 0.5 * a.pow(2), a - 0.5).mean(dim=reduce_dims)
    assert delta_flow.size(-1) == 2
    diff_y = distance_fn((delta_flow[:, :-1, :, :] - delta_flow[:, 1:, :, :]).abs())
    diff_x = distance_fn((delta_flow[:, :, :-1, :] - delta_flow[:, :, 1:, :]).abs())
    loss = diff_x + diff_y
    return loss


def flow_identity_loss(delta_flow):
    # Simply encourages the residual flow to be zero (L2):
    loss = delta_flow.pow(2).mean()
    return loss


def sample_gan_supervised_pairs(generator, ll, resize_fake2stn, psi, batch, dim_latent, freeze_ll, device, z=None):
    with torch.set_grad_enabled(not freeze_ll):
        if z is None:
            z = torch.randn(batch, dim_latent, device=device)
        unaligned_in, w_noise = generator([z], noise=None, return_latents=True)
        w_aligned = ll([w_noise[:, 0, :]], psi=psi)
        aligned_target, _ = generator(w_aligned, input_is_latent=True, noise=None)
        aligned_target = resize_fake2stn(aligned_target)
    return unaligned_in, aligned_target


def assign_fake_images_to_clusters(generator, stn, ll, loss_fn, resize_fake2stn, psi, batch, dim_latent, freeze_ll,
                                   num_heads, flips, device, sample_from_full_res=True, z=None, **stn_kwargs):
    """
    This function generates fake images, congeals them with the STN and then assigns the congealed images
    to their clusters.

    :return assignments_over_clusters_and_flips: a torch.min object with size (N,). Has indices and values fields to
                                                 access the results of the min operation.
            aligned_pred: (N*num_heads*(1+flips), C, flow_size, flow_size) The congealed output images output by the STN
            delta_flow: (N*num_heads*(1+flips), flow_size, flow_size, 2) The residual flow regressed by the STN
    """
    unaligned_in, aligned_target = sample_gan_supervised_pairs(generator, ll, resize_fake2stn, psi, batch, dim_latent,
                                                               freeze_ll, device, z)
    if flips:  # Try both the flipped and unflipped versions of unaligned_in, and eventually take the min of the two losses
        unaligned_in = torch.cat([unaligned_in, unaligned_in.flip(3, )], 0)  # (2 * N, C, H, W)
        aligned_target = aligned_target.repeat(2, 1, 1, 1)  # (2 * N, C, H, W)
        loss_size = (2, batch, num_heads)
    else:
        loss_size = (batch, num_heads)
    input_img_for_sampling = unaligned_in if sample_from_full_res else None
    resized_unaligned_in = resize_fake2stn(unaligned_in)
    aligned_pred, delta_flow = stn(resized_unaligned_in, return_flow=True,
                                   input_img_for_sampling=input_img_for_sampling, **stn_kwargs)
    perceptual_loss = loss_fn(aligned_pred, aligned_target).view(*loss_size)
    if flips:  # Merge cluster and flip dimensions:
        distance_collapsed = perceptual_loss.permute(1, 0, 2).reshape(batch, 2 * num_heads)  # (2, N, H) --> (N, 2 * H)
    else:
        distance_collapsed = perceptual_loss
    assignments_over_clusters_and_flips = distance_collapsed.min(dim=1)  # (N,)
    return assignments_over_clusters_and_flips, aligned_pred, delta_flow, unaligned_in, resized_unaligned_in, distance_collapsed


def gangealing_loss(generator, stn, ll, loss_fn, resize_fake2stn, psi, batch, dim_latent, freeze_ll, device, sample_from_full_res=False, **stn_kwargs):
    # The basic reconstruction loss used for GANgealing.
    # Important: Using a consistent set of noise images for both unaligned_in and
    # aligned_target surprisingly makes results much worse in some cases!
    # It turns out that it is actually better to have noise randomized between forward passes
    unaligned_in, aligned_target = sample_gan_supervised_pairs(generator, ll, resize_fake2stn, psi, batch, dim_latent,
                                                               freeze_ll, device, z=None)
    input_img_for_sampling = unaligned_in if sample_from_full_res else None
    aligned_pred, delta_flow = stn(resize_fake2stn(unaligned_in), return_flow=True,
                                   input_img_for_sampling=input_img_for_sampling, **stn_kwargs)
    perceptual_loss = loss_fn(aligned_pred, aligned_target).mean()
    return perceptual_loss, delta_flow


def gangealing_cluster_loss(generator, stn, ll, loss_fn, resize_fake2stn, psi, batch, dim_latent, freeze_ll, num_heads,
                            flips, device, sample_from_full_res=True, **stn_kwargs):
    # The reconstruction loss used in clustering variants of GANgealing.
    assignments, _, delta_flow, _, _, _ = \
        assign_fake_images_to_clusters(generator, stn, ll, loss_fn, resize_fake2stn, psi, batch, dim_latent, freeze_ll,
                                       num_heads, flips, device, sample_from_full_res, z=None, **stn_kwargs)
    assigned_perceptual_loss = assignments.values.mean()
    HW2 = delta_flow.size()[1:]  # delta_flow is of size (2 * N * args.num_heads, H, W, 2)
    if flips:  # Only the delta_flows corresponding to the assigned clusters get regularized:
        delta_flow = delta_flow.view(2, batch, num_heads, *HW2)  # (2, N, num_heads, H, W, 2)
        delta_flow = delta_flow.permute(1, 0, 2, 3, 4, 5).reshape(batch, 2 * num_heads, *HW2)
    else:
        delta_flow = delta_flow.view(batch, num_heads, *HW2)
    delta_flow = delta_flow[torch.arange(batch), assignments.indices]  # (N, H, W, 2)
    return assigned_perceptual_loss, delta_flow
