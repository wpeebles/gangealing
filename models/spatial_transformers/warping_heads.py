"""
Contains output heads useful for building Spatial Transformer Networks (STNs). These output heads
take an input image (and its features) as input, and then regress and apply a warp to the input image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.spatial_transformers.antialiased_sampling import MipmapWarp, Warp
from models.stylegan2.networks import EqualConv2d


class SimilarityHead(nn.Module):

    # Regresses and applies a similarity warp (rotation, uniform scale, vertical shift and horizontal shift)

    def __init__(self, in_shape, antialias=True, num_heads=1, **kwargs):
        """
        :param in_shape: int. SimiarlityHead.forward expects input features of shape (N, D). Pass D here.
        :param antialias: boolaen. Whether or not to use antialiasing when applying the similarity transformation.
        :param num_heads: int. Number of clusters being learned. Each cluster gets its own warping head.
        :param kwargs: Ignore; this is used to filter any parameters that are only processed by FlowHead
        """
        super().__init__()
        self.num_warp_params = 4  # rotation, uniform scale, horizontal shift, vertical shift
        self.linear = nn.Linear(in_shape, self.num_warp_params * num_heads, bias=True)
        # Initialize so linear always produces the identity transform in the first forward pass:
        self.linear.bias.data.zero_()
        self.linear.weight.data.zero_()
        # Select a pixel sampling algorithm:
        self.warper = MipmapWarp(max_num_levels=3.5) if antialias else Warp()
        self.num_heads = num_heads  # Number of clusters (each cluster gets its own warping head)
        self.register_buffer('one_hot', torch.tensor([0, 0, 1], dtype=torch.float).view(1, 1, 1, 3))

    @staticmethod
    def make_affine_matrix(rot, scale, shift_x, shift_y):
        # This function takes the raw output of the parameter regression network and converts them into
        # an affine matrix representing the predicted similarity transformation.
        # Inputs each of size (N, K), K = number of heads/clusters
        N, K = rot.size()
        rot = torch.tanh(rot) * math.pi
        scale = torch.exp(scale)
        cos_rot = torch.cos(rot)
        sin_rot = torch.sin(rot)
        matrix = [scale * cos_rot, -scale * sin_rot, shift_x,
                  scale * sin_rot, scale * cos_rot, shift_y]
        matrix = torch.stack(matrix, dim=2)  # (N, K, 6)
        matrix = matrix.reshape(N, K, 2, 3)  # (N, K, 2, 3)
        return matrix

    def make_3x3(self, M):
        # M should be size (N, K, 2, 3)
        one_hot = self.one_hot.expand(M.size(0), M.size(1), 1, 3)  # (N, K, 1, 3)
        M_3x3 = torch.cat([M, one_hot], 2)  # (N, K, 3, 3)
        return M_3x3

    def forward(self, img, features, output_resolution=None, alpha=None, base_warp=None,
                stop_grad=False, padding_mode='border', return_out_of_bounds=False, image_bounds=None,
                warp_policy='cartesian', unfold=False):
        """
        :param img: (N, C, H*, W*) input image that the STN will sample from when producing the warped output image.
        :param features: (N, D) features from which a similarity transformation will be predicted.
        :param output_resolution: int. The intermediate flow field will be bilinearly resampled to this resolution. If
                                  None, no resizing is performed on the flow.
        :param alpha: float in range [0, 1] (or None). If specified, linearly interpolates the predicted warp with the
                      identity transformation. This is mainly used for generating visuals, not training.
        :param base_warp: (N, K, 2, 3) tensor representing an affine warp (where K is num_heads). If specified, composes
                          the warp predicted by this function with base_warp.
        :param stop_grad: boolean. If True, gradients will not flow through the warp regression branch of the STN.
        :param padding_mode: ['border'/'reflection'/'zero']. Controls how the STN extrapolates pixels when sampling
                             beyond image boundaries.
        :param return_out_of_bounds: boolean. If True, checks if the predicted warp samples beyond the boundaries
                                     of the input image.
        :param image_bounds: (height, width) tuple of ints (or None). Useful for specifying the boundaries of the image
                             if using return_out_of_bounds. Otherwise not used.
        :param warp_policy: Only useful for clustering models (num_heads > 1). If 'cartesian', this function
                            will output K * N images (where K = num_heads). If torch.FloatTensor of size (N, K),
                            will output only N images, where the i-th input image is warped according to
                            torch.argmax(warp_policy, dim=1) (i.e., these should be scores for each input image
                            that represent the probability of being assigned to each cluster). Finally, if an nn.Module
                            is passed as this argument, it will be applied to img to produce the (N, K) score tensor.
        :param unfold: boolean. Only useful for clustering models. If True, output warped images will be of size
                       (N, K, C, H, W). Otherwise they will be of size (N*K, C, H, W) (where K = num_heads).
        :return: A tuple of outputs (K = num_heads):
                    (N * K, C, H, W) tensor, the warped output images.
                    (N * K, 2, 3) tensor, the predicted affine matrices.
                    (N * K, H, W, 2) tensor, the predicted similarity reverse sampling grid (flow)
                    (N * K,) boolean tensor (or None), whether each image exceeded image boundaries
        """
        N = features.size(0)
        params = self.linear(features)  # Regress raw similarity warp parameters from input features

        # This code block only pertains to clustering (num_heads > 1).
        # warp_policy == 'cartesian' --> warp every input input image with every STN head
        # warp_policy == 'assign_only' --> warp each input image with only one of the STN heads
        # For 'assign_only', we can pass-in raw classifier scores that we use to select the appropriate STN head
        # Alternatively, we can directly pass-in a classifier and compute them here for convenience.
        if isinstance(warp_policy, torch.Tensor):  # Pass externally-computed scores manually
            assignment_logits = warp_policy
            warp_policy = 'assign_only'
        elif isinstance(warp_policy, nn.Module):  # Pass an external classifier and compute the scores here
            assignment_logits = warp_policy(img)
            warp_policy = 'assign_only'
        else:
            assignment_logits = None
        if warp_policy == 'assign_only':  # Warp the i-th image according to its predicted cluster assignment
            predicted_assignments = assignment_logits.max(dim=1).indices
            assignments = predicted_assignments % self.num_heads  # The modulo handles flipping
            params = params.reshape(-1, self.num_warp_params, self.num_heads).permute(0, 2, 1)
            params = params.gather(1, assignments.view(N, 1, 1).repeat(1, 1, self.num_warp_params)).squeeze(1)
            split_size = 1
        elif warp_policy == 'cartesian':  # Warp all images with all warp heads
            split_size = self.num_heads
        else:
            raise NotImplementedError

        params = torch.split(params, split_size, dim=1)  # length-(num_warp_params) list of (N, split_size) tensors
        matrix = self.make_affine_matrix(*params)  # (N, K, 2, 3)
        if base_warp is not None:
            if base_warp.dim() == 3:
                base_warp = base_warp.unsqueeze(1)
            matrix = base_warp @ self.make_3x3(matrix)  # (N, K, 2, 3)
        if alpha is not None:  # NOTE: Currently, the same alpha is used for each head within a batch
            I = torch.eye(2, 3, device=matrix.device)[None, None]
            matrix = I.lerp(matrix, alpha[:, None, None, None])
        if output_resolution is None:
            img_size = torch.Size([img.size(0) * split_size, *img.size()[1:]])
        else:
            img_size = torch.Size([img.size(0) * split_size, img.size(1), output_resolution, output_resolution])
        if stop_grad:
            matrix = matrix.detach() + 0 * matrix  # DDP hack
        matrix = matrix.reshape(N * split_size, 2, 3)  # (N, split_size, 2, 3) --> (N * split_size, 2, 3)
        img = img.repeat_interleave(split_size, dim=0)  # (N, C, H, W) --> (N*split_size, C, H, W)
        grid = F.affine_grid(matrix, img_size, align_corners=False)
        out = self.warper(img, grid, padding_mode=padding_mode)  # (N*split_size, C, H, W)

        # This part is only used (optionally) for the automated data pre-processing application, not training.
        if return_out_of_bounds:
            oob = check_if_warp_exceeds_image_boundaries(grid, image_bounds, img_size, split_size)
        else:
            oob = None

        if unfold:
            out = out.reshape(N, -1, img_size[1], img_size[2], img_size[3])
            matrix = matrix.reshape(N, -1, 2, 3)
            grid = grid.reshape(N, -1, img_size[2], img_size[3], 2)
        return out, grid, matrix, oob


class FlowHead(nn.Module):

    # Regresses and applies an arbitrary transformation via reverse sampling

    def __init__(self, in_shape, antialias=True, num_heads=1, flow_downsample=8, **kwargs):
        super().__init__()
        self.flow_downsample = flow_downsample
        self.identity_flow = self.initialize_flow(in_shape).cuda()
        # This output head will produce a raw flow field at (128 / flow_downsample, 128 / flow_downsample) resolution:
        self.flow_out = nn.Sequential(EqualConv2d(in_shape[1], in_shape[1], 3, padding=1),
                                      nn.ReLU(),
                                      EqualConv2d(in_shape[1], num_heads * 2, 3, padding=1))
        # This ensures the output will initially be the identity transformation:
        nn.init.zeros_(self.flow_out[-1].weight)
        nn.init.zeros_(self.flow_out[-1].bias)
        # Mask head used for convex upsampling of the raw flow field (as done in RAFT)
        self.mask_out = nn.Sequential(EqualConv2d(in_shape[1], in_shape[1], 3, padding=1),
                                      nn.ReLU(),
                                      EqualConv2d(in_shape[1], num_heads * 9 * flow_downsample * flow_downsample, 3, padding=1))
        self.warper = MipmapWarp(max_num_levels=3.5) if antialias else Warp()
        self.num_heads = num_heads

    def initialize_flow(self, in_shape):
        N, C, H, W = in_shape
        # Identity sampling grid:
        coords = F.affine_grid(torch.eye(2, 3).unsqueeze(0),
                               (N, C, self.flow_downsample * H, self.flow_downsample * W))  # (N, H, W, 2)
        return coords

    def upsample_flow(self, flow, mask):  # RAFT: https://github.com/princeton-vl/RAFT/blob/master/core/update.py
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, H, W, _ = flow.shape
        flow = flow.permute(0, 3, 1, 2)  # NHW2 --> N2HW
        mask = mask.view(N, 1, 9, self.flow_downsample, self.flow_downsample, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(self.flow_downsample * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 4, 2, 5, 3, 1)
        up_flow = up_flow.reshape(N, self.flow_downsample*H, self.flow_downsample*W, 2)
        return up_flow

    def compute_flow(self, features):
        flow = self.flow_out(features)
        N, _, H, W = flow.size()
        flow = flow.reshape(N, self.num_heads, 2, H, W)
        flow = flow.permute(0, 1, 3, 4, 2)  # (N, K, H, W, 2)

        mask = self.mask_out(features)
        mask = mask.reshape(N, self.num_heads, 9 * self.flow_downsample * self.flow_downsample, H, W)
        return flow, mask

    def forward(self, img, features, output_resolution=None, alpha=None, base_warp=None,
                stop_grad=False, padding_mode='border', return_out_of_bounds=False, image_bounds=None,
                warp_policy='cartesian', unfold=False):
        """
        See SimilarityHead.forward's documentation above for a detailed description of input arguments (they are
        identical, other than the fact that features has spatial resolution here (of size (N, D, H*, W*)).
        :return: A tuple of outputs (K = num_heads):
                    (N * K, C, H, W) tensor, the warped output images.
                    (N * K, 128, 128, 2) tensor, the residual flow field predicted by this STN
                    (N * K, H*, W*, 2) tensor, the fully-composed flow field used for sampling the input image
                    (N * K,) boolean tensor (or None), whether each image exceeded image boundaries

        """
        low_res_delta_flow, mask = self.compute_flow(features)
        N, _, H, W, _ = low_res_delta_flow.size()

        if isinstance(warp_policy, torch.Tensor):  # Pass externally-computed logits manually
            assignment_logits = warp_policy
            warp_policy = 'assign_only'
        else:
            assignment_logits = None
        if warp_policy == 'assign_only':  # Warp the i-th image according to its predicted cluster assignment
            assignments = assignment_logits.max(dim=1).indices % self.num_heads  # The modulo handles external flipping
            batch_ix = torch.arange(N)
            low_res_delta_flow = low_res_delta_flow[batch_ix, assignments]
            mask = mask[batch_ix, assignments]
            split_size = 1
        elif warp_policy == 'cartesian':  # Warp all images according to all warp heads
            split_size = self.num_heads
        else:
            raise NotImplementedError

        low_res_delta_flow = low_res_delta_flow.reshape(N * split_size, H, W, 2)
        mask = mask.reshape(N * split_size, -1, H, W)
        delta_flow = self.upsample_flow(low_res_delta_flow, mask)
        flow = self.identity_flow + delta_flow
        if base_warp is not None:  # TODO: This currently assumes that base_warp is a similarity transform
            flow = apply_affine(base_warp, flow)
        if alpha is not None:  # NOTE: Currently, the same alpha is used for each head within a batch
            flow = self.identity_flow.lerp(flow, alpha[:, None, None, None])
        if output_resolution is None:
            img_size = torch.Size([img.size(0) * split_size, flow.size(1), flow.size(2)])
        else:
            img_size = torch.Size([img.size(0) * split_size, img.size(1), output_resolution, output_resolution])
            # interpolating the flow will yield a much higher quality output than interpolating pixels:
            flow = F.interpolate(flow.permute(0, 3, 1, 2), scale_factor=output_resolution / flow.size(2),
                                 mode='bilinear').permute(0, 2, 3, 1)
        if stop_grad:
            flow = flow.detach() + 0 * flow  # DDP hack
        img = img.repeat_interleave(split_size, dim=0)  # (N, C, H, W) --> (N*split_size, C, H, W)
        out = self.warper(img, flow, padding_mode=padding_mode)  # (N*split_size, C, H, W)
        # This part is only used (optionally) for the automated data pre-processing application, not training.
        if return_out_of_bounds:
            oob = check_if_warp_exceeds_image_boundaries(flow, image_bounds, img_size, split_size)
        else:
            oob = None
        if unfold:  # unfold=True should only be used when using warp_policy='cartesian' in ComposedSTN.forward
            out = out.reshape(out.size(0) // self.num_heads, self.num_heads, out.size(1), out.size(2), out.size(3))
            flow = flow.reshape(flow.size(0) // self.num_heads, self.num_heads, out.size(3), out.size(4), 2)
            delta_flow = delta_flow.reshape(delta_flow.size(0) // self.num_heads, self.num_heads, self.flow_downsample*H, self.flow_downsample*W, 2)
        return out, flow, delta_flow, oob


def apply_affine(matrix, grid):
    # This function is similar to torch.nn.functional.affine_grid, except it applies the
    # input affine matrix to an arbitrary input sampling grid instead of the identity sampling grid
    grid_size = grid.size()
    grid = grid.reshape(grid.size(0), -1, 2)
    ones = torch.ones(grid.size(0), grid.size(1), 1, device=grid.device)
    grid = torch.cat([grid, ones], 2)
    warped = grid @ matrix.permute(0, 2, 1)
    warped = warped.reshape(grid_size)
    return warped


def check_if_warp_exceeds_image_boundaries(grid, image_bounds, img_size, split_size, threshold=0.025):
    """
    This function checks if the warped image output by the STN exceeded the boundaries of the input image.
    :param grid: (N, H, W, 2) flow field predicted by the STN
    :param image_bounds: The height and width of the raw input image (before any border/zero padding).
                         Alternatively, this can be set as None, in which case this function checks to see
                         if the STN exceeded the boundaries of the input, pre-processed square image.
    :param img_size: Size of the transformed (warped) output image
    :param split_size: Number of outputs per input image (for non-clustering models this is always 1)
    :param threshold: The maximum number of pixels extrapolated by the STN for the input image to still be
                      considered "in-bounds."
    :return: (N,) boolean tensor. If the percentage of pixels in the output warped image that were sampled outside of
              image_bounds (i.e., extrapolated via border/reflection/zero/etc. padding) exceeds threshold, returns
              True. Otherwise returns False. This is computed per-input image in the batch.
    """
    if image_bounds is None:
        boundary_y = img_size[-2]
        boundary_x = img_size[-1]
    else:
        image_bounds = image_bounds.repeat_interleave(split_size, dim=0)  # TODO: test this part of the code
        landscape = image_bounds[:, 0] < image_bounds[:, 1]
        boundary_y = torch.where(landscape, img_size[-2] * image_bounds[:, 0] / image_bounds[:, 1],
                                 torch.tensor(img_size[-2], dtype=torch.float, device=grid.device)).round()
        boundary_x = torch.where(landscape, torch.tensor(img_size[-1], dtype=torch.float, device=grid.device),
                                 img_size[-1] * image_bounds[:, 1] / image_bounds[:, 0]).round()
    grid_x, grid_y = grid[..., 0], grid[..., 1]
    oob_x = grid_x.flatten(1).abs().gt((boundary_x - 1) / img_size[-1]).float().mean(dim=1).gt(threshold)
    oob_y = grid_y.flatten(1).abs().gt((boundary_y - 1) / img_size[-2]).float().mean(dim=1).gt(threshold)
    oob = torch.logical_or(oob_y, oob_x)
    return oob
