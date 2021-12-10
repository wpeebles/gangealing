"""
GANgealing training script.
"""

import torch
from torch import nn, optim
import numpy as np
from tqdm import tqdm
import json
import os

from models import Generator, get_stn, DirectionInterpolator, PCA, get_perceptual_loss, kmeans_plusplus, BilinearDownsample, accumulate, requires_grad
from models import gangealing_loss, gangealing_cluster_loss, total_variation_loss, flow_identity_loss
from datasets import img_dataloader, sample_infinite_data
from utils.vis_tools.training_vis import GANgealingWriter, create_training_visuals, create_training_cluster_visuals
from utils.distributed import all_gather, get_rank, setup_distributed, reduce_loss_dict, get_world_size, primary
from utils.base_argparse import base_training_argparse
from utils.annealing import DecayingCosineAnnealingWarmRestarts, lr_cycle_iters, get_psi_annealing_fn


def save_state_dict(ckpt_name, generator, t_module, t_ema, t_optim, t_sched, ll_module, ll_optim, ll_sched, args):
    ckpt_dict = {"g_ema": generator.state_dict(), "t": t_module.state_dict(),
                 "t_ema": t_ema.state_dict(), "t_optim": t_optim.state_dict(),
                 "t_sched": t_sched.state_dict(), "ll": ll_module.state_dict(),
                 "ll_optim": ll_optim.state_dict(), "ll_sched": ll_sched.state_dict(),
                 "args": args}
    torch.save(ckpt_dict, f'{results_path}/checkpoints/{ckpt_name}.pt')


def train(args, loader, generator, stn, t_ema, ll, t_optim, ll_optim, t_sched, ll_sched, loss_fn,
          anneal_fn, device, writer):

    # If using real data, select some fixed samples used to visualize training:
    vis_reals = loader is not None
    if vis_reals:
        if args.random_reals:
            real_indices = torch.randint(0, len(loader.dataset), (args.n_sample,)).numpy()
        else:
            real_indices = range(args.n_sample)
        sample_reals = torch.stack([loader.dataset[ix] for ix in real_indices]).to(device)
        loader = sample_infinite_data(loader, args.seed)
    else:
        sample_reals = None

    # Progress bar for monitoring training:
    pbar = range(args.iter)
    if primary():
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.2)

    # Record modules to make saving checkpoints easier:
    if args.distributed:
        t_module = stn.module
        ll_module = ll.module
    else:
        t_module = stn
        ll_module = ll

    sample_z = torch.randn(args.n_sample // args.num_heads, args.dim_latent, device=device)  # Used for generating a fixed set of GAN samples
    if args.clustering:  # A much larger fixed set of GAN samples used for generating clustering-specific visuals:
        big_sample_z = torch.randn(args.n_mean // get_world_size(), args.dim_latent, device=device)
    resize_fake2stn = BilinearDownsample(args.gen_size // args.flow_size, 3).to(device) if args.gen_size > args.flow_size else nn.Sequential()

    generator.eval()
    requires_grad(generator, False)  # G is frozen throughout this entire process
    requires_grad(stn, True)
    requires_grad(ll, True)

    # A model checkpoint will be saved whenever the learning rate is zero:
    zero_lr_iters = lr_cycle_iters(args.anneal_psi, args.period, args.iter, args.tm)
    early_ckpt_iters = set(zero_lr_iters)
    early_vis_iters = {100}
    early_vis_iters.update(early_ckpt_iters)

    # Initialize various training variables and constants:
    zero = torch.tensor(0.0, device='cuda')
    accum = 0.5 ** (32 / (10 * 1000))
    psi = 1.0  # initially there is no truncation

    # Create initial training visualizations:
    if args.clustering:
        create_training_cluster_visuals(generator, t_ema, ll, loss_fn, loader, resize_fake2stn, sample_z, big_sample_z,
                                        psi, device,  args.n_mean, args.n_sample, args.num_heads, args.flips,
                                        args.vis_batch_size, args.flow_size, 0, writer, padding_mode=args.padding_mode)
    else:
        create_training_visuals(generator, t_ema, ll, loader, sample_reals, resize_fake2stn, sample_z, psi, device,
                                args.n_mean, args.n_sample, 0, writer, padding_mode=args.padding_mode)

    for idx in pbar:  # main training loop
        i = idx + args.start_iter + 1
        if i <= args.anneal_psi:
            psi = anneal_fn(i, 1.0, 0.0, args.anneal_psi).item()
            psi_is_fixed = False
        else:
            psi = 0.0
            psi_is_fixed = True

        if i > args.iter:
            print("Done!")
            break

        ####################################
        ######### TRAIN STN and LL #########
        ####################################

        if args.clustering or args.flips:  # Clustering-specific perceptual loss:
            perceptual_loss, delta_flow = gangealing_cluster_loss(generator, stn, ll, loss_fn, resize_fake2stn, psi,
                                                                  args.batch, args.dim_latent, args.freeze_ll,
                                                                  args.num_heads, args.flips, device,
                                                                  sample_from_full_res=args.sample_from_full_res,
                                                                  padding_mode=args.padding_mode)
        else:  # Standard GANgealing perceptual loss (unimodal):
            perceptual_loss, delta_flow = gangealing_loss(generator, stn, ll, loss_fn, resize_fake2stn, psi,
                                                          args.batch, args.dim_latent, args.freeze_ll, device,
                                                          sample_from_full_res=args.sample_from_full_res,
                                                          padding_mode=args.padding_mode)
        tv_loss = total_variation_loss(delta_flow) if args.tv_weight > 0 else zero
        flow_idty_loss = flow_identity_loss(delta_flow) if args.flow_identity_weight > 0 else zero

        loss_dict = {}
        loss_dict["p"] = perceptual_loss
        loss_dict["tv"] = tv_loss
        loss_dict["f"] = flow_idty_loss

        stn.zero_grad()
        ll.zero_grad()
        full_stn_loss = perceptual_loss + args.tv_weight * tv_loss + args.flow_identity_weight * flow_idty_loss
        full_stn_loss.backward()
        t_optim.step()
        if not args.freeze_ll:
            ll_optim.step()
        if psi_is_fixed:  # Step learning rate schedulers once psi has been fully-annealed to zero:
            epoch = max(0, (i - args.anneal_psi) / args.period)
            t_sched.step(epoch)
            ll_sched.step(epoch)

        accumulate(t_ema, t_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)  # Aggregate loss information across GPUs

        if primary():
            # Display losses on the progress bar:
            perceptual_loss_val = loss_reduced["p"].mean().item()
            tv_loss_val = loss_reduced["tv"].mean().item()
            flow_idty_loss_val = loss_reduced["f"].mean().item()
            f_str = f"identity loss: {flow_idty_loss_val:.4f}; " if args.flow_identity_weight > 0 else ""
            tv_str = f"tv loss: {tv_loss_val:.6f}; " if args.tv_weight > 0 else ""
            pbar.set_description(f"perceptual loss: {perceptual_loss_val:.4f}; {tv_str}{f_str}psi: {psi:.4f}")

            # Log losses and others metrics to TensorBoard:
            if i % args.log_every == 0 or i in early_ckpt_iters:
                writer.add_scalar('Loss/Reconstruction', perceptual_loss_val, i)
                writer.add_scalar('Loss/TotalVariation', tv_loss_val, i)
                writer.add_scalar('Loss/FlowIdentity', flow_idty_loss_val, i)
                writer.add_scalar('Progress/psi', psi, i)
                writer.add_scalar('Progress/LL_LearningRate', ll_sched.get_last_lr()[0], i)
                writer.add_scalar('Progress/STN_LearningRate', t_sched.get_last_lr()[0], i)

            if i % args.ckpt_every == 0 or i in early_ckpt_iters:  # Save model checkpoint
                save_state_dict(str(i).zfill(7), generator, t_module, t_ema, t_optim, t_sched, ll_module,
                                ll_optim, ll_sched, args)

        if i % args.vis_every == 0 or i in early_vis_iters:  # Save visualizations to TensorBoard
            if primary() and i in early_ckpt_iters:
                pbar.write(f'{i:07}: Learning Rate = {t_sched.get_last_lr()[0]}')
            if args.clustering:
                create_training_cluster_visuals(generator, t_ema, ll, loss_fn, loader, resize_fake2stn, sample_z, big_sample_z,
                                                psi, device, args.n_mean, args.n_sample, args.num_heads, args.flips,
                                                args.vis_batch_size, args.flow_size, i, writer,
                                                padding_mode=args.padding_mode)
            else:
                create_training_visuals(generator, t_ema, ll, loader, sample_reals, resize_fake2stn, sample_z, psi,
                                        device, args.n_mean, args.n_sample, i, writer, padding_mode=args.padding_mode)


if __name__ == "__main__":
    device = "cuda"
    parser = base_training_argparse()
    args = parser.parse_args()
    if args.transform == 'similarity':
        assert args.tv_weight == 0, 'Total Variation loss is not currently supported for similarity-only STNs'
    args.n_mean = 200 if args.debug else args.n_mean
    args.vis_batch_size //= args.num_heads  # Keep visualization batch size reasonable for clustering models
    # Setup distributed PyTorch and create results directory:
    args.distributed = setup_distributed(args.local_rank)
    args.clustering = args.num_heads > 1
    results_path = os.path.join(args.results, args.exp_name)
    if primary():
        writer = GANgealingWriter(results_path)
        with open(f'{results_path}/opt.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else:
        writer = None

    # Seed RNG:
    torch.manual_seed(args.seed * get_world_size() + get_rank())
    np.random.seed(args.seed * get_world_size() + get_rank())

    # Initialize models:
    generator = Generator(args.gen_size, args.dim_latent, args.n_mlp, channel_multiplier=args.gen_channel_multiplier, num_fp16_res=args.num_fp16_res).to(device)
    stn = get_stn(args.transform, flow_size=args.flow_size, supersize=args.real_size, channel_multiplier=args.stn_channel_multiplier, num_heads=args.num_heads).to(device)
    t_ema = get_stn(args.transform, flow_size=args.flow_size, supersize=args.real_size, channel_multiplier=args.stn_channel_multiplier, num_heads=args.num_heads).to(device)
    ll = DirectionInterpolator(pca_path=None, n_comps=args.ndirs, inject_index=args.inject, n_latent=generator.n_latent, num_heads=args.num_heads).to(device)
    accumulate(t_ema, stn, 0)

    # Setup optimizers and learning rate schedulers:
    t_optim = optim.Adam(stn.parameters(), lr=args.stn_lr, betas=(0.9, 0.999), eps=1e-8)
    ll_optim = optim.Adam(ll.parameters(), lr=args.ll_lr, betas=(0.9, 0.999), eps=1e-8)
    t_sched = DecayingCosineAnnealingWarmRestarts(t_optim, T_0=1, T_mult=args.tm, decay=args.decay)
    ll_sched = DecayingCosineAnnealingWarmRestarts(ll_optim, T_0=1, T_mult=args.tm, decay=args.decay)

    # Setup the perceptual loss function:
    loss_fn = get_perceptual_loss(args.loss_fn, device)
    # Get the function used to anneal psi:
    anneal_fn = get_psi_annealing_fn(args.anneal_fn)

    # Load pre-trained generator (and optionally resume from a GANgealing checkpoint):
    print(f"Loading model from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"])  # NOTE: We load g_ema as generator since G is frozen!
    try:  # Restore from full checkpoint, including the optimizer
        stn.load_state_dict(ckpt["t"])
        t_ema.load_state_dict(ckpt["t_ema"])
        t_optim.load_state_dict(ckpt["t_optim"])
        t_sched.load_state_dict(ckpt["t_sched"])
        ll.load_state_dict(ckpt["ll"])
        ll_optim.load_state_dict(ckpt["ll_optim"])
        ll_sched.load_state_dict(ckpt["ll_sched"])
    except KeyError:
        print('Only G_EMA has been loaded from checkpoint. Other nets are random!')
        n_pca = 1000 if args.debug else 1000000
        with torch.no_grad():
            batch_w = generator.batch_latent(n_pca // get_world_size())
        batch_w = all_gather(batch_w)
        pca = PCA(args.ndirs, batch_w)
        ll.assign_buffers(pca)
        if args.clustering:  # For clustering models, initialize using K-Means++ on W-Space
            print('Running K-Means++ Initialization')
            if args.debug:
                centroids = generator.batch_latent(args.num_heads).detach().requires_grad_(False)
            else:
                centroids = kmeans_plusplus(args.num_heads, 50000, generator, loss_fn, args.inject)
            decomposed = pca.encode(centroids)
            ll.assign_coefficients(decomposed)

    # See if the start iteration can be recovered when resuming training:
    args.start_iter = 0
    try:
        ckpt_name = os.path.basename(args.ckpt)
        if ckpt_name.startswith('best_'):
            ckpt_name = ckpt_name[5:]  # Remove prefix
        args.start_iter = int(os.path.splitext(ckpt_name)[0])
    except ValueError:
        pass

    # Move models to DDP if distributed training is enabled:
    if args.distributed:
        stn = nn.parallel.DistributedDataParallel(stn, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
        ll = nn.parallel.DistributedDataParallel(ll, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)

    # Setup real data for visualizations (optional):
    loader = img_dataloader(args.real_data_path, shuffle=False, batch_size=args.vis_batch_size, resolution=args.real_size, infinite=False) if args.real_data_path is not None else None

    # Begin training:
    train(args, loader, generator, stn, t_ema, ll, t_optim, ll_optim, t_sched, ll_sched, loss_fn, anneal_fn,
          device, writer)
