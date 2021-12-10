import argparse
import torch
from models import get_stn, ResnetClassifier
from utils.download import find_model


def base_eval_argparse():
    parser = argparse.ArgumentParser(description="Use Pre-Trained GANgealing Checkpoints")
    # loading the Spatial Transformer:
    parser.add_argument("--ckpt", type=str, required=True, help="path to GANgealing checkpoint")
    parser.add_argument("--transform", default=['similarity', 'flow'], choices=['similarity', 'flow'], nargs='+', type=str, help='Which class of warps the STN is constrained to produce. Default: most expressive.')
    parser.add_argument("--flow_size", type=int, default=128, help="resolution of the flow fields learned by the STN")
    parser.add_argument("--stn_channel_multiplier", type=int, default=0.5, help='controls the number of channels in the STN\'s convolutional layers')
    parser.add_argument("--num_heads", default=1, type=int, help='Number of clusters learned by the STN')
    # Spatial Transformer forward pass hyperparameters:
    parser.add_argument("--iters", default=1, type=int, help='Number of times to recursively run the similarity STN')
    parser.add_argument("--padding_mode", default='border', choices=['border', 'zeros', 'reflection'], type=str, help='Padding algorithm for when the STN samples beyond image boundaries')
    parser.add_argument("--no_flip_inference", action='store_true', help='If specified, no horizontal flips will be attempted. Only affects non-clustering models (num_heads==1)')
    # loading data:
    parser.add_argument("--real_data_path", type=str, default=None, help="Path to real data")
    parser.add_argument("--real_size", default=256, type=int, help='resolution of real images')
    parser.add_argument("--batch", type=int, default=50, help="batch size per-GPU for evaluation")
    parser.add_argument("--seed", default=0, type=int, help='Random seed for evaluation')
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")

    return parser


def load_stn(args, load_classifier=False, device='cuda'):
    try:
        supersize = args.crop_size
    except:
        supersize = args.real_size
    ckpt = find_model(args.ckpt)
    t_ema = get_stn(args.transform, flow_size=args.flow_size, supersize=supersize,
                    channel_multiplier=args.stn_channel_multiplier, num_heads=args.num_heads).to(device)
    t_ema.load_state_dict(ckpt['t_ema'])
    t_ema.eval()  # The STN doesn't use eval-specific ops, so this shouldn't do anything
    if load_classifier:  # Also return the cluster classifier if it exists:
        if 'classifier' in ckpt:
            classifier = ResnetClassifier(args.flow_size, channel_multiplier=args.stn_channel_multiplier,
                                          num_heads=2 * args.num_heads, supersize=supersize).to(device)
            classifier.load_state_dict(ckpt['classifier'])
            classifier.eval()  # Shouldn't do anything
            return t_ema, classifier
        else:
            return t_ema, None
    else:
        return t_ema


def determine_flips(args, t, classifier, input_imgs, cluster=None, return_cluster_assignments=False):
    # There are two ways a flip can be done with GANgealing at test time:
    # (1) For clustering models, directly predict if an input image should be flipped using the cluster classifier net
    # (2) In general, try running both img and flip(img) through the STN. Decide to flip img based on which of the two
    #     produces the smoothest residual flow field.
    # This function predicts the flip using method (1) if classifier is supplied; otherwise uses method (2).
    if classifier is not None:  # predict the flip using the clustering classifier directly:
        if cluster is None:  # infer the cluster using the classifier
            data_flipped, _, clusters, flip_indices = classifier.run_flip(input_imgs)
            clusters = clusters % args.num_heads
        else:  # use the passed cluster
            data_flipped, flip_indices = classifier.run_flip_target(input_imgs, cluster)
            clusters = torch.tensor(cluster, dtype=torch.long, device='cuda').repeat(input_imgs.size(0))
        warp_policy = torch.eye(args.num_heads, device=input_imgs.device)[clusters]
    elif not args.no_flip_inference:  # automatically infer which images need to be horizontally flipped/mirrored:
        _, data_flipped, flip_indices = t.forward_with_flip(input_imgs, return_inputs=True, return_flip_indices=True,
                                                            padding_mode=args.padding_mode, iters=args.iters)
        warp_policy = 'cartesian'
        clusters = torch.zeros(input_imgs.size(0), dtype=torch.long, device=input_imgs.device)
    else:  # Don't perform any flipping:
        data_flipped = input_imgs
        flip_indices = torch.zeros(input_imgs.size(0), 1, 1, 1, device=input_imgs.device, dtype=torch.bool)
        warp_policy = 'cartesian'
        clusters = torch.zeros(input_imgs.size(0), dtype=torch.long, device=input_imgs.device)
    if return_cluster_assignments:
        return data_flipped, flip_indices, warp_policy, clusters
    else:
        return data_flipped, flip_indices, warp_policy
