import argparse


def base_training_argparse():
    parser = argparse.ArgumentParser(description="GANgealing Training")
    # Main training arguments:
    parser.add_argument("--exp-name", type=str, required=True, help="Name for experiment run (used for logging)")
    parser.add_argument("--ckpt", type=str, required=True, help="path to either a StyleGAN2(-ADA) generator checkpoint or path to a previous GANgealing checkpoint to resume training from")
    parser.add_argument("--load_G_only", action='store_true', help="If specified, will only load g_ema from the input checkpoint (otherwise, will attempt to load the full GANgealing checkpoint)")
    parser.add_argument("--dim_latent", type=int, default=512, help="dimensionality of W-Space")
    parser.add_argument("--n_mlp", type=int, default=8, help="number of linear layers in the mapping network")
    parser.add_argument("--gen_channel_multiplier", type=int, default=2, help="channel multiplier for the generator")
    parser.add_argument("--num_fp16_res", type=int, default=0, help="number of final generator layers that will use mixed precision. For StyleGAN2-ADA checkpoints, this should be set to 4 (including --cfg=stylegan2 checkpoints trained in the ADA codebase)")
    parser.add_argument("--results", type=str, default='results', help='path to the results directory')
    parser.add_argument("--seed", default=0, type=int, help='Random seed for this experiment')
    parser.add_argument("--real_data_path", type=str, default=None, help="Path to real data (used ONLY for making visualizations, not training)")
    parser.add_argument("--real_size", default=256, type=int, help='resolution of real images')
    parser.add_argument("--gen_size", default=256, type=int, help='resolution of fake images produced by the generator')
    parser.add_argument("--iter", type=int, default=800000, help="total training iterations")
    parser.add_argument("--batch", type=int, default=5, help="batch size per-GPU")
    parser.add_argument("--debug", action='store_true', help='If specified, quickly begins training for faster debugging')

    # GANgealing hyperparameters:
    parser.add_argument("--inject", default=5, type=int, help='The index of the last W+ StyleGAN input that will receive the learned congealing vector as input. Later layers = stronger alignment effect, but also potentially more challenging optimization')
    parser.add_argument("--ndirs", default=1, type=int, help='Number of W-Space PCA coefficients learned. A larger number gives more flexibility to the STN, but also potentially yields more degenerate solutions.')
    parser.add_argument("--anneal_psi", default=150000, type=int, help='Number of iterations over which psi should be annealed from 1 to 0')
    parser.add_argument("--anneal_fn", type=str, choices=['cosine', 'linear'], default='cosine', help='Controls the scheduler for annealing psi from 1 to 0 at the beginning of training')
    parser.add_argument("--loss_fn", type=str, default='vgg_ssl', choices=['lpips', 'vgg_ssl'], help='The perceptual loss to use. Note that vgg_ssl is fully-unsupervised (backbone VGG trained with SimCLR)')
    parser.add_argument("--tv_weight", default=1000.0, type=float, help='Loss weighting of the Total Variation smoothness regularizer on the residual flow')
    parser.add_argument("--flow_identity_weight", default=1.0, type=float, help='Loss weighting of the identity regularizer on the residual flow')
    parser.add_argument("--freeze_ll", action='store_true', help='If specified, disables learning of the congealing vector')
    parser.add_argument("--sample_from_full_res", action='store_true', help='If specified, pixels will be sampled from the full resolution fake images during training instead of from downsampled versions')

    # Clustering hyperparameters (leave default to use standard unimodal GANgealing):
    parser.add_argument("--num_heads", default=1, type=int, help='The number of clusters to learn and independently congeal. Setting >1 enables the clustering version of GANgealing')
    parser.add_argument("--flips", action='store_true', help='If specified, during training input fake images and their mirrors are input to the STN, and only the min of the two losses is optimized')

    # Model hyperparameters:
    parser.add_argument("--transform", default=['similarity', 'flow'], choices=['similarity', 'flow'], nargs='+', type=str, help='The class of warps the STN is constrained to produce. Default: most expressive.')
    parser.add_argument("--padding_mode", default='reflection', choices=['border', 'zeros', 'reflection'], type=str, help='Padding algorithm for when the STN samples beyond image boundaries')
    parser.add_argument("--stn_lr", type=float, default=0.001, help="base learning rate of SpatialTransformer")
    parser.add_argument("--ll_lr", type=float, default=0.01, help="base learning rate of latent congealing code")
    parser.add_argument("--flow_size", type=int, default=128, help="resolution of the flow fields learned by the STN")
    parser.add_argument("--stn_channel_multiplier", type=int, default=0.5, help='controls the number of channels in the STN\'s convolutional layers')

    # Visualization hyperparameters:
    parser.add_argument("--vis_every", type=int, default=5000, help='frequency with which visualizations are generated during training')
    parser.add_argument("--ckpt_every", type=int, default=50000, help='frequency of checkpointing during training')
    parser.add_argument("--log_every", default=25, type=int, help='How frequently to log data to TensorBoard')
    parser.add_argument("--n_mean", type=int, default=8000, help='number of real images used to generate average image visualizations periodically during training')
    parser.add_argument("--n_sample", type=int, default=64, help="number of images (real and fake) to generate visuals for")
    parser.add_argument("--vis_batch_size", default=250, type=int, help='batch size used to generate visuals')
    parser.add_argument("--random_reals", action='store_true', help='If specified, visualization will be performed on a random set of real images instead of the first N in the dataset')

    # Learning Rate scheduler hyperparameters:
    parser.add_argument("--period", default=37500, type=float, help='Period for cosine learning rate scheduler (measured in gradient steps)')
    parser.add_argument("--decay", default=0.9, type=float, help='Decay factor for the cosine learning rate scheduler')
    parser.add_argument("--tm", default=2, type=int, help='Period multiplier for the cosine learning rate scheduler')

    return parser
