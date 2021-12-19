"""
GANgealing cluster classifier training script. This script is only relevant for the clustering versions of GANgealing.

The point of this script is to train a classifier that takes as input an image and predicts two things:
(1) which of our learned clusters that image should be assigned to
(2) whether or not the image should be flipped before being processed by its assigned STN (if --flips is specified)
"""

import torch
from torch import nn, optim
import numpy as np
from tqdm import tqdm
import json
import os

from models import Generator, get_stn, DirectionInterpolator, ResnetClassifier, get_perceptual_loss, BilinearDownsample, requires_grad
from models import assign_fake_images_to_clusters, accuracy
from datasets import img_dataloader
from utils.vis_tools.training_vis import GANgealingWriter, create_training_cluster_visuals, create_training_cluster_classifier_visuals
from utils.distributed import get_rank, setup_distributed, reduce_loss_dict, get_world_size, primary
from utils.base_argparse import base_training_argparse
from utils.annealing import DecayingCosineAnnealingWarmRestarts, lr_cycle_iters


def save_state_dict(ckpt_name, classifier, generator, t_ema, ll, cls_optim, cls_sched, args):
    ckpt_dict = {"classifier": classifier.state_dict(), "g_ema": generator.state_dict(),
                 "t_ema": t_ema.state_dict(), "ll": ll.state_dict(), "cls_optim": cls_optim.state_dict(),
                 "cls_sched": cls_sched.state_dict(), "args": args}
    torch.save(ckpt_dict, f'{results_path}/checkpoints/{ckpt_name}.pt')


def train(args, loader, classifier, generator, t_ema, ll, cls_optim, cls_sched, loss_fn, device, writer):

    # Progress bar for monitoring training:
    pbar = range(args.iter)
    if primary():
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.2)

    # Record modules to make saving checkpoints easier:
    if args.distributed:
        classifier_module = classifier.module
    else:
        classifier_module = classifier

    sample_z = torch.randn(args.n_sample // args.num_heads, args.dim_latent, device=device)  # Used for generating a fixed set of GAN samples
    big_sample_z = torch.randn(args.n_mean // get_world_size(), args.dim_latent, device=device)
    resize_fake2stn = BilinearDownsample(args.gen_size // args.flow_size, 3).to(device) if args.gen_size > args.flow_size else nn.Sequential()

    generator.eval()
    requires_grad(generator, False)  # G is frozen throughout this entire process
    requires_grad(t_ema, False)
    requires_grad(ll, False)
    requires_grad(classifier, True)

    # A model checkpoint will be saved whenever the learning rate is zero:
    zero_lr_iters = lr_cycle_iters(args.anneal_psi, args.period, args.iter, args.tm)
    early_ckpt_iters = set(zero_lr_iters)
    early_vis_iters = {100}
    early_vis_iters.update(early_ckpt_iters)

    psi = 0.0
    total_num_clusters = args.num_heads * (1 + args.flips)  # if we are flipping, there are 2X as many clusters to predict
    xent = nn.CrossEntropyLoss().to(device)

    # Create initial training visualizations:
    create_training_cluster_visuals(generator, t_ema, ll, loss_fn, loader, resize_fake2stn, sample_z, big_sample_z,
                                    psi, device,  args.n_mean, args.n_sample, args.num_heads, args.flips,
                                    args.vis_batch_size, args.flow_size, 0, writer, padding_mode=args.padding_mode)
    create_training_cluster_classifier_visuals(t_ema, classifier, loader, args.num_heads, args.n_mean, args.n_sample,
                                               device, 0, writer, padding_mode=args.padding_mode)

    for idx in pbar:  # main training loop
        i = idx + args.start_iter + 1

        if i > args.iter:
            print("Done!")
            break

        ####################################
        ##### TRAIN CLUSTER CLASSIFIER #####
        ####################################
        # Sample a batch of fake images and figure out which clusters they belong to. These pairs
        # of (fake image, cluster index) are the training data for our cluster classifier.
        with torch.no_grad():  # No need to backprop through any of the image formation/ cluster assignment process:
            assignments_over_clusters_and_flips, _, _, unaligned_in, resized_unaligned_in, distance = \
                assign_fake_images_to_clusters(generator, t_ema, ll, loss_fn, resize_fake2stn, psi, args.batch,
                                               args.dim_latent, True, args.num_heads, args.flips, device,
                                               sample_from_full_res=args.sample_from_full_res, z=None,
                                               padding_mode=args.padding_mode)
        predicted_assignments = classifier(resized_unaligned_in[:args.batch])
        xent_loss = xent(predicted_assignments, assignments_over_clusters_and_flips.indices)
        acc1 = accuracy(predicted_assignments, -distance)
        acc2 = accuracy(predicted_assignments, -distance, k=2)  # "reverse" top-K accuracy, K=2 (see accuracy docs)

        loss_dict = {"cross_entropy": xent_loss, "acc@1": acc1, "acc@2": acc2}
        assignments_per_head = torch.bincount(assignments_over_clusters_and_flips.indices, minlength=total_num_clusters).div(float(args.batch))
        pred_assignments_per_head = torch.bincount(predicted_assignments.argmax(dim=1), minlength=total_num_clusters).div(float(args.batch))
        for cluster_ix, (num_assignments_gt, num_assignments_pred) in enumerate(
                zip(assignments_per_head, pred_assignments_per_head)):
            loss_dict[f"head_{cluster_ix}"] = num_assignments_gt
            loss_dict[f"pred_head_{cluster_ix}"] = num_assignments_pred

        classifier.zero_grad()
        xent_loss.backward()
        cls_optim.step()
        cls_sched.step(i / args.period)

        loss_reduced = reduce_loss_dict(loss_dict)  # Aggregate loss information across GPUs

        if primary():
            # Display losses on the progress bar:
            xent_val = loss_reduced["cross_entropy"].mean().item()
            acc1_val = loss_reduced["acc@1"].mean().item()
            acc2_val = loss_reduced["acc@2"].mean().item()
            pbar.set_description(f"cross entropy: {xent_val:.4f}; top-1 acc: {acc1_val:.4f}; top-2 acc: {acc2_val:.4f}")

            # Log loss metrics to TensorBoard:
            writer.add_scalar('Loss/CrossEntropy', xent_val, i)
            writer.add_scalar('Loss/Accuracy@1', acc1_val, i)
            writer.add_scalar('Loss/Accuracy@2', acc2_val, i)
            writer.add_scalars('Loss/AssignmentsGT',
                               {f'head_{head_ix}': loss_reduced[f'head_{head_ix}'].mean().item() for head_ix in
                                range(total_num_clusters)}, i)
            writer.add_scalars('Loss/AssignmentsPredicted',
                               {f'head_{head_ix}': loss_reduced[f'pred_head_{head_ix}'].mean().item() for head_ix in
                                range(total_num_clusters)}, i)
            writer.add_scalar('Progress/LearningRate', cls_sched.get_last_lr()[0], i)

            if i % args.ckpt_every == 0 or i in early_ckpt_iters:  # Save model checkpoint
                save_state_dict(str(i).zfill(7), classifier_module, generator, t_ema, ll, cls_optim, cls_sched, args)

        if i % args.vis_every == 0 or i in early_vis_iters:  # Save visualizations to TensorBoard
            if primary() and i in early_ckpt_iters:
                pbar.write(f'{i:07}: Learning Rate = {cls_sched.get_last_lr()[0]}')
            create_training_cluster_classifier_visuals(t_ema, classifier, loader, args.num_heads, args.n_mean,
                                                       args.n_sample,
                                                       device, i, writer, padding_mode=args.padding_mode)


if __name__ == "__main__":
    device = "cuda"

    parser = base_training_argparse()
    parser.add_argument("--cls_lr", default=0.001, type=float, help='base learning rate of cluster classifier')
    args = parser.parse_args()

    args.anneal_psi = 0
    args.n_mean = 200 if args.debug else args.n_mean
    args.vis_batch_size //= args.num_heads  # Keep visualization batch size reasonable for clustering models
    # Setup distributed PyTorch and create results directory:
    args.distributed = setup_distributed()
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
    generator = Generator(args.gen_size, args.dim_latent, args.n_mlp, channel_multiplier=args.gen_channel_multiplier).to(device)
    t_ema = get_stn(args.transform, flow_size=args.flow_size, supersize=args.real_size, channel_multiplier=args.stn_channel_multiplier, num_heads=args.num_heads).to(device)
    ll = DirectionInterpolator(pca_path=None, n_comps=args.ndirs, inject_index=args.inject, n_latent=generator.n_latent, num_heads=args.num_heads).to(device)
    # Note: no batch norm/dropout/etc. used, so no need to worry about eval versus training mode:
    classifier = ResnetClassifier(args.flow_size, channel_multiplier=args.stn_channel_multiplier,
                                  num_heads=args.num_heads * (1 + args.flips), supersize=args.real_size).to(device)

    # Setup optimizers and learning rate schedulers:
    cls_optim = optim.Adam(classifier.parameters(), lr=args.cls_lr)
    cls_sched = DecayingCosineAnnealingWarmRestarts(cls_optim, T_0=1, T_mult=args.tm, decay=args.decay)
    # Setup the perceptual loss function:
    loss_fn = get_perceptual_loss(args.loss_fn, device)

    # Load pre-trained STN, Generator and LL (required):
    print(f"Loading model from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"])
    t_ema.load_state_dict(ckpt["t_ema"])
    ll.load_state_dict(ckpt["ll"])
    # We initialize the classifier with the similarity STN's weights to speed-up training:
    assert args.transform[0] == 'similarity'
    if len(args.transform) == 1:
        classifier.load_state_dict(ckpt["t_ema"], strict=False)
    else:  # Load only the similarity network
        classifier.load_state_dict(t_ema.stns[0].state_dict(), strict=False)
    args.start_iter = 0
    try:  # Try to resume training of the cluster classifier:
        classifier.load_state_dict(ckpt["classifier"])
        cls_optim.load_state_dict(ckpt["cls_optim"])
        cls_sched.load_state_dict(ckpt["cls_sched"])
        print('Resuming cluster classifier training.')
        try:  # See if the start iteration can be recovered:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except ValueError:
            pass
    except KeyError:
        pass

    # Move cluster classifier to DDP if distributed training is enabled:
    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        classifier = nn.parallel.DistributedDataParallel(classifier, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    # Setup real data for visualizations:
    loader = img_dataloader(args.real_data_path, shuffle=False, batch_size=args.vis_batch_size, resolution=args.real_size)

    # Begin training the cluster classifier:
    train(args, loader, classifier, generator, t_ema, ll, cls_optim, cls_sched, loss_fn, device, writer)
