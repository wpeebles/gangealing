"""
This script runs a pre-trained Spatial Transformer on an input dataset and records the smoothness of the flow field
produced by the STN for every image. These smoothness values are treated as scores which can be used to filter the
dataset. An image with low (highly negative) smoothness corresponds to an image that should be removed.
"""

import torch
from torch.utils.data import Subset
from tqdm import tqdm
from models import total_variation_loss
from applications import base_eval_argparse, load_stn, determine_flips
from utils.distributed import setup_distributed, synchronize, all_gather, primary
from datasets import img_dataloader
import os


def get_flow_scores(args, path, t):
    flow_score_path = f'{path}/flow_scores.pt'
    if os.path.exists(flow_score_path):  # Return a cached copy of flow scores
        return torch.load(flow_score_path)
    else:  # Compute and cache flow scores:
        return compute_flow_scores(args, t)


@torch.inference_mode()
def compute_flow_scores(args, t):
    loader = img_dataloader(args.real_data_path, resolution=args.real_size, batch_size=args.batch, shuffle=False,
                            distributed=args.distributed, infinite=False)
    num_total = len(loader.dataset)
    scores = []
    pbar = tqdm(loader) if primary() else loader
    for batch in pbar:
        batch = batch.to('cuda')
        batch, _, _ = determine_flips(args, t, None, batch)
        _, flows = t(batch, return_flow=True, iters=args.iters, padding_mode=args.padding_mode)
        smoothness = total_variation_loss(flows, reduce_batch=False)
        scores.append(smoothness)
    scores = -torch.cat(scores, 0)  # lower (more negative) scores indicate worse images
    synchronize()
    scores = all_gather(scores, cat=False)
    scores = scores.permute(1, 0).reshape(-1)[:num_total]
    if primary():
        score_path = f'{args.real_data_path}/flow_scores.pt'
        torch.save(scores.cpu(), score_path)
        print(f'num_scores = {scores.size(0)}')
        print(f'Flow scores saved at {score_path}')
    return scores


def get_high_score_indices(scores, fraction_retained):
    q = 1 - fraction_retained
    min_score = torch.quantile(scores, q)
    high_score_indices, = torch.where(scores > min_score)
    return high_score_indices.tolist()


def filter_dataset(dataset, scores, fraction_retained):
    """
    This function removes.
    :param dataset: PyTorch Dataset instance to filter
    :param scores: 1D tensor of scores, with same size as dataset or a path to the scores
    :param fraction_retained: float between 0 and 1, the fraction of images from dataset to retain. The images with
                              lowest scores will be dropped.
    :return: PyTorch Dataset instance with lowest scoring images removed from the dataset
    """
    if isinstance(scores, str):
        scores = torch.load(scores)
    high_score_indices = get_high_score_indices(scores, fraction_retained)
    filtered_dataset = Subset(dataset, high_score_indices)
    return filtered_dataset


if __name__ == '__main__':
    parser = base_eval_argparse()
    args = parser.parse_args()
    assert args.num_heads == 1, 'Clustering not currently supported for flow_scores.py'
    args.distributed = setup_distributed()
    t_ema = load_stn(args)
    compute_flow_scores(args, t_ema)
