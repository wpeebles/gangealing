import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.stylegan2.networks import EqualLinear, ConvLayer, ResBlock
from models.spatial_transformers.warping_heads import SimilarityHead, FlowHead
from models.spatial_transformers.antialiased_sampling import BilinearDownsample
from models.losses.loss import total_variation_loss


def get_stn(transforms, **stn_kwargs):
    is_str = isinstance(transforms, str)
    is_list = isinstance(transforms, list)
    assert is_str or is_list
    if is_str:
        transforms = [transforms]
    if len(transforms) == 1:
        return SpatialTransformer(transform=transforms[0], **stn_kwargs)
    else:
        return ComposedSTN(transforms, **stn_kwargs)


def unravel_index(indices, shape):
    # https://stackoverflow.com/a/65168284
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord, dim=-1)

    return coord


class ComposedSTN(nn.Module):

    """
    Chains a sequence of STNs together by composing warps. This module provides some connective tissue
    to let STNs that perform different warps (e.g., similarity and per-pixel flow) talk with each other.

    This class has only been tested with transforms=['similarity', 'flow']. Chaining multiple unique similarity STNs
    (optionally followed by a final flow STN) should likely work as well. A flow STN followed by a similarity STN is not
    currently supported and will not work without some light modifications to SimilarityHead. Similarly, chaining
    multiple unique flow STNs together will not work without some modifications to FlowHead.
    """

    def __init__(self, transforms, **stn_kwargs):
        super().__init__()
        stns = []
        for transform in transforms:
            stns.append(SpatialTransformer(transform=transform, **stn_kwargs))
        if transforms != ['similarity', 'flow']:
            print('WARNING: ComposedSTN is only tested for transforms=["similarity", "flow"].')
        self.stns = nn.ModuleList(stns)
        self.transforms = transforms[:]  # Deepcopy
        self.stn_in_size = stn_kwargs['flow_size']
        self.N_minus_1 = len(self.stns) - 1  # Number of STNs minus 1
        self.is_flow = 'flow' in transforms
        if self.is_flow:
            self.identity_flow = self.stns[transforms.index('flow')].identity_flow
        self.num_heads = self.stns[0].warp_head.num_heads
        if self.num_heads > 1:
            self.cluster_assignments = torch.eye(self.num_heads)  # (K, K)

    def forward(self, input_img, return_warp=None, return_flow=False, return_sim=False, return_intermediates=False,
                output_resolution=None, unfold=False, iters=1, alpha=None, warp_policy='cartesian',
                input_img_for_sampling=None, **stn_forward_kwargs):
        """
        :param return_warp: If True, returns the final (N, H, W, 2) sampling grid produced by the composed STN
        :param return_flow: If True, returns either the final (N, K, 2, 3) affine matrix (if the final network is a
                            similarity STN) or the final (N, H, W, 2) residual flow (if the final network is a flow STN)
        :param return_sim: If True, returns the output image produced by the very first of the composed STNs
        :param return_intermediates: If True, returns a list of intermediate output images from each STN and a list
                                     of (N, H, W, 2) sampling grids that were used to form each intermediate image
        :param stn_forward_kwargs: Any additional arguments that should be used in each STN's forward pass
        :return: (N, C, H, W) output warped images, as well as any additional requested outputs

        See spatial_transformers.warping_heads.SimilarityHead for documentation of the other inputs.

        """
        out = input_img
        source_pixels = input_img if input_img_for_sampling is None else input_img_for_sampling
        warp = None  # None corresponds to identity warp
        intermediate_output_resolution = self.stn_in_size
        intermediate_imgs, intermediate_warps = [], []
        N = source_pixels.size(0)
        for i, stn in enumerate(self.stns):
            alpha_t = alpha if i == self.N_minus_1 else None
            output_resolution_t = output_resolution if i == self.N_minus_1 else intermediate_output_resolution
            unfold_t = unfold if i == self.N_minus_1 else False
            iters_t = iters if i == 0 else 1  # Currently, only iterating the similarity transform is supported
            if self.num_heads > 1 and warp_policy == 'cartesian':  # Handle clustering stuff
                if i == 0:
                    warp_policy_t = warp_policy
                else:
                    warp_policy_t = self.cluster_assignments.to(source_pixels.device).repeat(N, 1)
            else:
                warp_policy_t = warp_policy

            stn_out = stn(out, return_warp=True, return_flow=True, return_intermediates=False,
                          input_img_for_sampling=source_pixels, base_warp=warp,
                          output_resolution=output_resolution_t, unfold=unfold_t,
                          iters=iters_t, alpha=alpha_t, warp_policy=warp_policy_t, **stn_forward_kwargs)
            out, grid, flow_or_matrix = stn_out
            if self.num_heads > 1 and warp_policy == 'cartesian' and i == 0:
                source_pixels = source_pixels.repeat_interleave(self.num_heads, dim=0)
            if return_intermediates:
                intermediate_imgs.append(out)
                intermediate_warps.append(grid)
            if return_sim and i == 0:
                sim_out = out
            # TODO: Currently, flow --> similarity and flow --> flow is not supported
            # warp = flow if self.transforms[i + 1] == 'similarity' else grid
            warp = flow_or_matrix
        if return_intermediates:
            return intermediate_imgs, intermediate_warps
        ret = [out]
        if return_warp:
            ret.append(grid)
        if return_flow:
            ret.append(flow_or_matrix)
        if return_sim:
            ret.append(sim_out)  # TODO: make this less hard-coded (assumes first transform is similarity)
        if len(ret) == 1:
            ret = ret[0]
        return ret

    def uncongeal_points(self, imgB, points_congealed, output_resolution=None, iters=1, unnormalize_output_points=True,
                         normalize_input_points=False, return_congealed_img=False, **stn_forward_kwargs):
        """
        Given input images imgA, transfer known key points from the congealed space to imgA.
        """
        assert imgB.size(0) == points_congealed.size(0)
        if normalize_input_points:
            points_congealed = SpatialTransformer.normalize(points_congealed, imgB.size(-1), self.stn_in_size)
        congealed_img, gridB = self.forward(imgB, return_warp=True, output_resolution=output_resolution, iters=iters,
                                            **stn_forward_kwargs)
        pointsB = F.grid_sample(gridB.permute(0, 3, 1, 2), points_congealed.unsqueeze(2).float(), padding_mode='border').squeeze(3).permute(0, 2, 1)
        if unnormalize_output_points:
            pointsB = SpatialTransformer.unnormalize(pointsB, imgB.size(-1), imgB.size(-1))  # Back to coordinates in [0, H-1]
        if return_congealed_img:
            return pointsB, congealed_img
        else:
            return pointsB

    def congeal_points(self, imgA, pointsA, output_resolution=None, iters=1, normalize_input_points=True,
                          unnormalize_output_points=False, return_full=False, **stn_forward_kwargs):
        """
        Given input images imgA, transfer known key points from the congealed space to imgA.
        """
        assert imgA.size(0) == pointsA.size(0)
        intermediate_output_resolution = self.stn_in_size
        outA = imgA
        points_congealed = pointsA
        warpA = None
        for i, stn in enumerate(self.stns):  # Compose in forward order
            output_resolution_t = output_resolution if i == self.N_minus_1 else intermediate_output_resolution
            norm_input_points = normalize_input_points if i == 0 else True
            unnorm_out_points = unnormalize_output_points if i == self.N_minus_1 else True
            iters_i = iters if i == 0 else 1
            outA, warpA, points_congealed = stn.congeal_points(outA, points_congealed, normalize_input_points=norm_input_points,
                                                           unnormalize_output_points=unnorm_out_points, iters=iters_i,
                                                           output_resolution=output_resolution_t, base_warp=warpA,
                                                           input_img_for_sampling=imgA, return_full=True,
                                                           **stn_forward_kwargs)
        if return_full:
            return outA, warpA, points_congealed
        else:
            return points_congealed

    def transfer_points(self, imgA, imgB, pointsA, output_resolution=None, iters=1, congeal_kwargs={}, uncongeal_kwargs={},
                        **stn_forward_kwargs):
        """
        Given input images imgA, transfer known key points pointsA to target images imgB.
        """
        assert imgA.size(0) == imgB.size(0) == pointsA.size(0)
        # Step 1: Map the key points in imgA to the congealed image (forward warp):
        points_congealed = self.congeal_points(imgA, pointsA, output_resolution=output_resolution,
                                               normalize_input_points=True, iters=iters, **congeal_kwargs,
                                               **stn_forward_kwargs)
        # Step 2: Map the key points in the congealed image to those in imgB (reverse warp):
        pointsB = self.uncongeal_points(imgB, points_congealed, output_resolution=output_resolution,
                                        normalize_input_points=True, unnormalize_output_points=True,
                                        iters=iters, **uncongeal_kwargs, **stn_forward_kwargs)
        return pointsB

    def forward_with_flip(self, input_img, return_flow=False, return_warp=False, return_inputs=False,
                          return_flip_indices=False, **stn_forward_kwargs):
        """
        This function runs both input_img as well as mirror(input_img) through the Spatial Transformer. The smoothness
        of the resulting flows gives a strong indication whether input_img should be mirrored or not. So this function
        returns T(input_img) if its flow field is smoother than T(mirror(input_img)), and vice versa.
        :param input_img: (N, C, H, W) tensor of input images
        :param return_flow: If True, returns the residual flow field (N, H, W, 2)
        :param return_warp: If True, returns the full flow field (N, H, W, 2)
        :param return_inputs: If True, returns a (N, C, H, W) tensor of inputs (flipped where needed)
        :param return_flip_indices: If True, returns a boolean tensor of size (N, 1, 1, 1) indicating which images were
                                    mirrored
        :param stn_forward_kwargs: Any additional arguments for ComposedSTN.forward
        :return: (N, C, H*, W*) tensor of congealed output images, where the i-th batch element will be
                 T(input_img[i]) if the flow field is smoother than that of T(mirror(input_img[i])), and vice versa.
                 The same applies for return_flow, return_warp, return_inputs and return_flip_indices above.
        """
        congealed, warp, flow = self.forward(input_img, return_warp=True, return_flow=True, **stn_forward_kwargs)
        input_imgF = input_img.flip(3,)
        congealedF, warpF, flowF = self.forward(input_imgF, return_warp=True, return_flow=True,
                                                **stn_forward_kwargs)
        smoothness = total_variation_loss(flow, reduce_batch=False)
        smoothnessF = total_variation_loss(flowF, reduce_batch=False)
        mirror_indicator = torch.stack([smoothness, smoothnessF], 0).argmin(dim=0).view(input_img.size(0), 1, 1, 1).bool()
        congealed_out = torch.where(mirror_indicator, congealedF, congealed)
        out = [congealed_out]
        if return_warp:
            warpF[:, :, :, 0] = -warpF[:, :, :, 0]
            warp_out = torch.where(mirror_indicator, warpF, warp)
            out.append(warp_out)
        if return_flow:
            flow_out = torch.where(mirror_indicator, flowF, flow)
            out.append(flow_out)
        if return_inputs:
            input_flip = torch.where(mirror_indicator, input_imgF, input_img)
            out.append(input_flip)
        if return_flip_indices:
            out.append(mirror_indicator)
        if len(out) == 1:
            out = out[0]
        return out

    def match_flows(self, imgA, imgB, pointsA, pointsB=None, permutation=None, **stn_forward_kwargs):
        """
        This function tries to align the flows produced by the inputs imgA and imgB by horizontally-flipping
        one, both or neither image (determined batch-wise between each pair). When it does perform a flip, pointsA is
        updated accordingly such that correspondence is maintained horizontally. This essentially does the same
        thing as forward_with_flip except it returns mirrored *input* images and updates key point annotations.

        :param imgA: (N, C, H, W) input tensor of images
        :param imgB: (N, C, H, W) input tensor of images
        :param pointsA: (N, num_points, *) input tensor of points corresponding to imgA (* = 2 or 3)
        :param pointsB: (N, num_points, *) input tensor of points corresponding to imgB
        :param permutation: List of length (num_points) indicating how key point labels should change when flipped
        :return: imgA: (N, C, H, W) tensor where some images may be horizontally-flipped
                 imgB: (N, C, H, W) tensor where some images may be horizontally-flipped
                 pointsA: (N, num_points, *) tensor where pointsA[i, :, 0] will be updated if imgA[i] was flipped
                 pointsB: (N, num_points, *) tensor where pointsB[i, :, 0] will be updated if imgB[i] was flipped (optional)
                 pick: (N, 1, 1, 1) tensor where pick[i] is one of {0, 1, 2, 3}:
                    pick[i] == 0 if imgA[i] and imgB[i] were both NOT flipped
                    pick[i] == 1 if imgA[i] was flipped but imgB[i] was NOT flipped
                    pick[i] == 2 if imgA[i] was NOT flipped but imgB[i] was flipped
                    pick[i] == 3 if imgA[i] and imgB[i] were both flipped
        """
        # flows have dimension (N, H, W, 2), so we will sum over everything but the batch dimension:
        # x.flip(3,) == horizontally-flipped x:
        imgA_flip, imgB_flip = imgA.flip(3,), imgB.flip(3,)
        inputs = torch.cat([imgA, imgB, imgA_flip, imgB_flip], 0)
        _, flows = self.forward(inputs, return_flow=True, **stn_forward_kwargs)
        flowA, flowB, flowAf, flowBf = flows.chunk(4, dim=0)
        tvA = total_variation_loss(flowA, reduce_batch=False)
        tvAf = total_variation_loss(flowAf, reduce_batch=False)
        tvB = total_variation_loss(flowB, reduce_batch=False)
        tvBf = total_variation_loss(flowBf, reduce_batch=False)
        A2B = tvA + tvB
        Af2B = tvAf + tvB
        A2Bf = tvA + tvBf
        Af2Bf = tvAf + tvBf
        pick = torch.stack([A2B, Af2B, A2Bf, Af2Bf], 0).argmin(dim=0).view(imgA.size(0), 1, 1, 1)  # Pick the best match per-batch element
        imgA = torch.where(pick % 2 == 0, imgA, imgA_flip)
        imgB = torch.where(pick <= 1, imgB, imgB_flip)
        # If we flipped imgA[i], then we need to update the x-coordinate of its key points pointsA[i] as well:
        pointsA = pointsA.clone()
        pointsA[:, :, 0] = torch.where((pick % 2 == 0).view(pick.size(0), 1), pointsA[:, :, 0],
                                   imgA.size(-1) - 1 - pointsA[:, :, 0])
        if permutation is not None:
            pointsA = torch.where((pick % 2 == 0).view(pick.size(0), 1, 1), pointsA, pointsA[:, permutation])
        if pointsB is not None:
            pointsB = pointsB.clone()
            pointsB[:, :, 0] = torch.where((pick <= 1).view(pick.size(0), 1), pointsB[:, :, 0],
                                       imgB.size(-1) - 1 - pointsB[:, :, 0])
            if permutation is not None:
                pointsA = torch.where((pick <= 1).view(pick.size(0), 1, 1), pointsA, pointsA[:, permutation])
            return imgA, imgB, pointsA, pointsB, pick
        else:
            return imgA, imgB, pointsA, pick

    def propagate_object(self, congealed_object_points, congealed_object_values, congealed_mask_values, target_image,
                         sigma, cluster_classifier=None, cluster=None, mem_efficient=False, **uncongeal_kwargs):
        from utils.splat2d_cuda import splat2d

        # (1) Sanity-check inputs:
        device = congealed_object_points.device
        assert congealed_object_points.size(0) == congealed_mask_values.size(0) == target_image.size(0) == sigma.size(0), \
            'all tensor inputs should have the same batch size'
        N = congealed_object_points.size(0)
        supersize = target_image.size(-1)
        assert supersize == target_image.size(-2), 'This function currently only supports square input images'
        assert congealed_object_points.dim() == congealed_mask_values.dim() == 3

        # (2) For clustering models, determine if we need to flip or not:
        if self.num_heads == 1:  # For K=1 GANgealing things are simple:
            warp_policy = 'cartesian'
            flip = torch.zeros(N, device=device, dtype=torch.bool)
        else:  # For K>1 we need to figure out which cluster we are propagating from:
            assert cluster_classifier is not None, \
                'A cluster_classifier needs to be supplied since this is a clustering model'
            warp_policy = torch.eye(self.num_heads, device=device)[cluster].unsqueeze_(0).repeat(N, 1)
            flip = cluster_classifier.run_flip_target(target_image, cluster)
        flip = flip.view(N, 1, 1, 1)

        # (3) Propagate the object points to target_image:
        propagated_points = self.uncongeal_points(target_image, congealed_object_points, normalize_input_points=False,
                                                     unnormalize_output_points=True, warp_policy=warp_policy,
                                                     **uncongeal_kwargs)  # (N, num_points, 2)

        # (4) Determine which points are visible (within bounds of target_image).
        # NOTE: The final image will use the unrounded (potentially sub-pixel) points later on.
        _rounded = propagated_points.round()  # We only round to verify points are in-bounds
        in_bound_points = (_rounded[:, :, 0] >= 0).logical_and(_rounded[:, :, 1] >= 0).logical_and \
                          (_rounded[:, :, 0] < supersize).logical_and(_rounded[:, :, 1] < supersize)
        valid_point_indices = [torch.where(in_bound_point_i)[0] for in_bound_point_i in in_bound_points]
        num_valid_points = [indices.size(0) for indices in valid_point_indices]

        # (5) Splat the object points into the target_image coordinate system. Splatting can be interpreted as a
        # "soft" version of torch.scatter that handles sub-pixel indices.
        # See if every batch element has the same number of visible points:
        batch_splat = num_valid_points == [num_valid_points[0]] * N and not mem_efficient
        blank_img = torch.zeros_like(target_image)  # (N, C, H, W)
        blank_mask = blank_img[:, :1]  # (N, 1, H, W)
        if batch_splat:  # If there are the same number of points in every image, then we can splat in batch mode:
            valid_point_indices = torch.stack(valid_point_indices).unsqueeze(2)  # (N, num_valid_points, 1)
            prop_points = propagated_points.gather(dim=1, index=valid_point_indices.repeat(1, 1, 2))  # (N, num_valid_points, 2)
            object_values = congealed_object_values.gather(dim=1, index=valid_point_indices.repeat(1, 1, 3))  # (N, num_valid_points, 3)
            mask_values = congealed_mask_values.gather(dim=1, index=valid_point_indices)  # (N, num_valid_points, 1)
            propagated_object_img = splat2d(blank_img, prop_points, object_values, sigma, False)  # (N, C, H, W)
            propagated_mask_img = splat2d(blank_mask, prop_points, mask_values, sigma, True)  # (N, 1, H, W)
        else:  # Otherwise, we need to fall-back to the slower online splatting:
            blank_img = blank_img[:1]  # (1, C, H, W)
            blank_mask = blank_img[:, :1]  # (1, 1, H, W)
            propagated_object_img = []
            propagated_mask_img = []
            for i in range(N):  # Iterate over batch dimension
                valid_point_indices_i = valid_point_indices[i]  # (num_valid_points,)
                prop_points = propagated_points[i: i + 1, valid_point_indices_i]  # (1, num_valid_points, 2)
                object_values = congealed_object_values[i: i + 1, valid_point_indices_i]  # (1, num_valid_points, 3)
                mask_values = congealed_mask_values[i: i + 1, valid_point_indices_i]  # (1, num_valid_points, 1)
                prop_obj_img = splat2d(blank_img, prop_points, object_values, sigma[i: i + 1], False)  # (1, C, H, W)
                prop_mask_img = splat2d(blank_mask, prop_points, mask_values, sigma[i: i + 1], True)  # (1, 1, H, W)
                propagated_object_img.append(prop_obj_img)
                propagated_mask_img.append(prop_mask_img)
            propagated_object_img = torch.cat(propagated_object_img, 0)  # (N, C, H, W)
            propagated_mask_img = torch.cat(propagated_mask_img, 0)  # (N, 1, H, W)
        # (6) Perform horizontal flips if predicted by the cluster classifier:
        propagated_object_img = torch.where(flip, propagated_object_img.flip(3,), propagated_object_img)  # (N, C, H, W)
        propagated_mask_img = torch.where(flip, propagated_mask_img.flip(3,), propagated_mask_img)  # (N, 1, H, W)
        return propagated_object_img, propagated_mask_img

    def load_single_state_dict(self, state_dict, index, strict=True):
        # state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith(f'stns.{index}')}
        # print(state_dict.keys())
        return self.stns[index].load_state_dict(state_dict, strict)

    def load_several_state_dicts(self, state_dicts, indices, strict=True):
        assert len(state_dicts) == len(indices)
        for state_dict, index in zip(state_dicts, indices):
            self.load_single_state_dict(state_dict, index, strict)

    def load_state_dict(self, state_dict, strict=True):
        ignore = ['warp_head.one_hot']
        for i in range(len(self.stns)):
            ignore.extend([f'stns.{i}.input_downsample.kernel_horz', f'stns.{i}.input_downsample.kernel_vert',
                           f'stns.{i}.warp_head.rebias'])
        ignore = set(ignore)
        filtered = {k: v for k, v in state_dict.items() if k not in ignore}
        return super().load_state_dict(filtered, False)


class SpatialTransformer(nn.Module):
    def __init__(self, flow_size, supersize, channel_multiplier=0.5, blur_kernel=[1, 3, 3, 1],
                 num_heads=1, transform='similarity', flow_downsample=8):
        """
        Here is how the SpatialTransformer works:
            (1) Take an image as input.
            (2) Regress a flow field from this input image at some fixed resolution (flow_size, flow_size),
                usually flow_size=128.
            (3) Optionally upsample/downsample this flow field with bilinear interpolation. Note that bilinear
                 upsampling the flow field will usually result in a very high quality output image (much higher quality
                 than bilinearly resizing the warped output image after-the-fact).
            (4) Sample the input image with the (optionally resized) flow field to obtain the warped output image.

        :param flow_size: The resolution of the flow field produced by the Spatial Transformer and also the resolution
                      at which images are processed by the warp parameter regression portion of the STN.
        :param supersize: The resolution of input images to the Spatial Transformer. Should be >= flow_size.
        :param channel_multiplier: Controls the number of channels in the conv layers
        :param blur_kernel: Low-pass filter to lightly anti-alias intermediate activations of the network.
        :param num_heads: Number of clusters (and thus warping heads)
        :param transform: Class of transformations produced by the STN (either 'similarity' or 'flow')
        :param flow_downsample: Only relevant when transform = 'flow'. As part of Step (2) above, a low resolution
                flow is initially regressed at resolution (flow_size / flow_downsample, flow_size / flow_downsample),
                before being upsampled to (flow_size, flow_size) via learned convex upsampling as described in the RAFT
                paper. Should be a power of 2.
        """
        super().__init__()

        if supersize > flow_size:
            self.input_downsample = BilinearDownsample(supersize // flow_size, 3)

        self.input_downsample_required = supersize > flow_size
        self.stn_in_size = flow_size
        self.is_flow = transform == 'flow'

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, int(channels[flow_size]), 1)]

        log_size = int(math.log(flow_size, 2))
        log_downsample = int(math.log(flow_downsample, 2))

        in_channel = channels[flow_size]

        end_log = log_size - 4 if self.is_flow else 2
        assert end_log >= 0

        num_downsamples = 0
        for i in range(log_size, end_log, -1):
            downsample = (not self.is_flow) or (num_downsamples < log_downsample)
            num_downsamples += downsample
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(int(in_channel), int(out_channel), blur_kernel, downsample))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        if not self.is_flow:
            self.final_linear = EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu')

        if transform == 'similarity':
            warp_class = SimilarityHead
            in_shape = channels[4]
        elif transform == 'flow':
            warp_class = FlowHead
            in_shape = (1, in_channel, flow_size // flow_downsample, flow_size // flow_downsample)
        else:
            raise NotImplementedError
        self.warp_head = warp_class(in_shape, antialias=True, num_heads=num_heads, flow_downsample=flow_downsample)
        if self.is_flow:
            self.identity_flow = self.warp_head.identity_flow

    def forward(self, input_img, output_resolution=None, iters=1, return_warp=False, return_flow=False,
                return_intermediates=False, return_out_of_bounds=False, intermediate_output_resolution=None,
                stop_grad=False, alpha=None, padding_mode='border', input_img_for_sampling=None, image_bounds=None,
                warp_policy='cartesian', unfold=False, base_warp=None):
        """
        :param input_img: (N, C, H, W) input image tensor used to regress a warp. If input_img_for_sampling (see below)
                           is NOT specified, then input_img will also be used as the source for sampling pixels
                           according the STN's predicted warp.
        :param output_resolution: int (or None). This will be the size of the output warped image and the predicted flow
                                  field. Internally, this bilinearly resizes the flow field, and thus yields a much
                                  higher quality warped output image compared to bilinear resizing in pixel space.
        :param iters: int, number of times to iterate the STN on its own output (composing warps each time).
                      Currently, only similarity STNs support iters > 1.
        :param return_warp: If True, returns the final (composed) (N, H, W, 2) sampling grid regressed by the STN
        :param return_flow: This argument's behavior varies depending on the type of WarpHead. If this is a similarity
                            STN, this will return an (N, 2, 3) tensor of affine matrices representing the similarity
                            transformation. Otherwise, if this is an unconstrained flow STN, this will return an
                            (N, H, W, 2) tensor representing the _residual_ flow field predicted by the STN.
        :param return_intermediates: If True, returns intermediate output images and warps at each iteration of the STN
        :param return_out_of_bounds: If True, returns (N,) boolean tensor indicating if the STN sampled beyond image
                                     boundaries.
        :param intermediate_output_resolution: int (or None)The size of the intermediate warped images produced by the
                                                STN. None = most efficient resolution automatically selected.
        :param input_img_for_sampling: (N, C, H*, W*) input image tensor. If specified, the STN will sample from this
                                        image instead of input_img above. This argument is useful if, e.g., you have a
                                        high resolution version of input_img; then you can pass the high resolution
                                        image here to get a higher quality output warped image.

        For explanations of all other inputs, please refer to the documentation of warping_heads.SimilarityHead.forward.

        :return: (N, C, H, W) tensor of congealed output images if unfold=False. Otherwise, (N, K, C, H, W) tensor is
                 returned, where K is the number of clusters. Additional outputs will be returned if return_warp,
                 return_flow, return_intermediates or return_out_of_bounds is True.
        """

        if iters == 1:  # Apply the STN just a single time to input_img:
            return self.single_forward(input_img, output_resolution=output_resolution, return_warp=return_warp,
                                       return_flow=return_flow, stop_grad=stop_grad, alpha=alpha,
                                       padding_mode=padding_mode, input_img_for_sampling=input_img_for_sampling,
                                       return_out_of_bounds=return_out_of_bounds, image_bounds=image_bounds,
                                       warp_policy=warp_policy, unfold=unfold, base_warp=base_warp)
        else:  # Continually apply the STN to its own output, composing the predicted warps at each iteration:
            return self.iterated_forward(input_img, output_resolution=output_resolution, iters=iters,
                                         return_warp=return_warp, return_flow=return_flow,
                                         return_intermediates=return_intermediates,
                                         intermediate_output_resolution=intermediate_output_resolution,
                                         stop_grad=stop_grad, alpha=alpha, padding_mode=padding_mode,
                                         input_img_for_sampling=input_img_for_sampling,
                                         return_out_of_bounds=return_out_of_bounds, image_bounds=image_bounds,
                                         warp_policy=warp_policy, unfold=unfold, base_warp=base_warp)

    def iterated_forward(self, input_img, output_resolution=None, iters=1, return_warp=False, return_flow=False,
                         return_intermediates=False, intermediate_output_resolution=None, stop_grad=False, alpha=None,
                         padding_mode='border', input_img_for_sampling=None, return_out_of_bounds=False, image_bounds=None,
                         warp_policy='cartesian', unfold=False, base_warp=None):

        # Please see SpatialTransformer.forward above for documentation regarding these inputs.
        assert not self.is_flow, 'iterated_forward is currently only supported for similarity STNs'
        out = input_img
        source_pixels = input_img if input_img_for_sampling is None else input_img_for_sampling
        if intermediate_output_resolution is None:
            intermediate_output_resolution = self.stn_in_size
        M = base_warp
        if return_intermediates:
            outs = []
            transforms = []
        for it in range(iters):
            last_iter = it == (iters - 1)
            output_resolution_t = output_resolution if last_iter else intermediate_output_resolution
            alpha_t = alpha if last_iter else None
            return_oob_t = return_out_of_bounds and last_iter
            unfold_t = unfold and last_iter
            itr_out = self.single_forward(out, output_resolution=output_resolution_t, return_warp=True, return_flow=True,
                                          return_out_of_bounds=return_oob_t, base_warp=M,
                                          input_img_for_sampling=source_pixels, stop_grad=stop_grad, alpha=alpha_t,
                                          padding_mode=padding_mode, image_bounds=image_bounds, warp_policy=warp_policy,
                                          unfold=unfold_t, pack=True)
            out, grid, M, oob = itr_out
            if return_oob_t:
                out_of_bounds = oob
            if return_intermediates:
                outs.append(out)
                transforms.append(M)

        if return_intermediates:
            return outs, transforms
        rtn = [out]
        if return_warp:  # TODO: could probably make argument-packing more elegant:
            rtn.append(grid)
        if return_flow:
            rtn.append(M)
        if return_out_of_bounds:
            rtn.append(out_of_bounds)
        if len(rtn) == 1:
            rtn = rtn[0]
        return rtn

    def single_forward(self, input_img, output_resolution=None, return_warp=False, return_flow=False,
                       return_out_of_bounds=False, base_warp=None, input_img_for_sampling=None, stop_grad=False,
                       alpha=None, padding_mode='border', image_bounds=None, warp_policy='cartesian', unfold=False,
                       pack=False):

        # Please see SpatialTransformer.forward above for documentation regarding all of these inputs.
        """
        :param pack: boolean, if True returns all outputs from the WarpHead instance.
        """

        if input_img.size(-1) > self.stn_in_size:
            regression_input = self.input_downsample(input_img)
        else:
            regression_input = input_img

        if input_img_for_sampling is not None:
            source_pixels = input_img_for_sampling
        else:
            source_pixels = input_img

        out = self.convs(regression_input)

        batch, channel, height, width = out.shape
        out = self.final_conv(out)

        if not self.is_flow:
            out = out.view(batch, -1)
            out = self.final_linear(out)

        output_resolution = output_resolution if output_resolution is not None else self.stn_in_size
        out, grid, M, oob = self.warp_head(source_pixels, out, output_resolution=output_resolution, base_warp=base_warp,
                                           stop_grad=stop_grad, alpha=alpha, padding_mode=padding_mode,
                                           return_out_of_bounds=return_out_of_bounds, image_bounds=image_bounds,
                                           warp_policy=warp_policy, unfold=unfold)
        if pack:  # Return everything, even if oob is None
            return [out, grid, M, oob]
        else:  # TODO: could make argument-packing more elegant:
            rtn = [out]
            if return_warp:
                rtn.append(grid)
            if return_flow:
                rtn.append(M)
            if return_out_of_bounds:
                rtn.append(oob)
            if len(rtn) == 1:
                rtn = rtn[0]
            return rtn

    @staticmethod
    def normalize(points, res, out_res):
        return points.div(out_res - 1).add(-0.5).mul(2).mul((res - 1) / res)

    @staticmethod
    def unnormalize(points, res, out_res):
        return points.div((res - 1) / res).div(2).add(0.5).mul(out_res - 1)

    @staticmethod
    def convert(points, current_res, target_res):
        points = SpatialTransformer.normalize(points, target_res, current_res)
        points = SpatialTransformer.unnormalize(points, target_res, target_res)
        return points

    def congeal_points(self, imgA, pointsA, normalize_input_points=True, unnormalize_output_points=False, output_resolution=None,
                       iters=1, input_img_for_sampling=None, return_full=False, **stn_forward_kwargs):
        assert imgA.size(0) == pointsA.size(0)
        N = imgA.size(0)
        num_points = pointsA.size(1)
        source_res = imgA.size(-1) if input_img_for_sampling is None else input_img_for_sampling.size(-1)
        outA, gridA, flow_or_matrixA = self.forward(imgA, return_warp=True, return_flow=True,
                                                    output_resolution=output_resolution,
                                                    input_img_for_sampling=input_img_for_sampling,
                                                    iters=iters, **stn_forward_kwargs)
        if normalize_input_points:
            pointsA = self.normalize(pointsA, source_res, source_res)  # [0, H-1] --> [-1, 1]
        # Forward similarity transform (nice and closed-form):
        if not self.is_flow:
            pointsA = torch.cat([pointsA, torch.ones(N, num_points, 1, device=pointsA.device)], 2)  # (N, |points|, 3)
            onehot = torch.tensor([[[0, 0, 1]]], dtype=torch.float, device=flow_or_matrixA.device).repeat(N, 1, 1)
            matrixA_3x3 = torch.cat([flow_or_matrixA, onehot], 1)
            A2congealed = torch.inverse(matrixA_3x3).permute(0, 2, 1)
            points_congealed = (pointsA @ A2congealed)[..., [0, 1]]  # Apply transformation and remove homogeneous coordinate
            if unnormalize_output_points:
                points_congealed = self.unnormalize(points_congealed, source_res, source_res)
        # While we have access to the reverse (inverse) flows since our STN using reverse sampling, we do NOT have
        # access to the forward flows. To approximate the A --> congealed forward flow,
        # we roughly reverse the congealed --> A inverse flow (which we have access to) via a brute-force nearest
        # neighbor search on the sampling grid.
        else:
            assert flow_or_matrixA.size(-1) == 2  # flow_or_matrixA is a flow field with shape (N, H, W, 2)
            gridA = flow_or_matrixA + self.identity_flow
            gridA_reshaped = gridA.reshape(N, gridA.size(1), gridA.size(2), 1, 1, 2)  # (N, H, W, 1, 1, 2)
            pointsA = pointsA.reshape(N, 1, 1, num_points, 2, 1)  # (N, 1, 1, num_points, 2, 1)
            similarities = (gridA_reshaped @ pointsA)[..., 0, 0]  # (N, H, W, num_points)
            # Compute distances as: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
            distances = pointsA.pow(2).squeeze(-1).sum(dim=-1) + gridA_reshaped.pow(2).sum(dim=-1).squeeze(-1) - 2 * similarities
            # TODO: currently, key points that get forward mapped beyond the congealed image boundary are clamped to
            # TODO: the border. There's probably a better way to handle this problem...
            nearest_neighbors = distances.reshape(N, gridA_reshaped.size(1) * gridA_reshaped.size(2), num_points).argmin(
                dim=1)  # (N, num_points)
            points_congealed = unravel_index(nearest_neighbors, (gridA_reshaped.size(1), gridA_reshaped.size(2)))  # (N, num_points, 2)
        if return_full:
            return outA, flow_or_matrixA, points_congealed
        else:
            return points_congealed

    def uncongeal_points(self, imgB, points_congealed, unnormalize_output_points=True, normalize_input_points=False,
                         output_resolution=None, iters=1, input_img_for_sampling=None, **stn_forward_kwargs):
        """
        Given input images imgB, transfer known key points points_congealed to target images imgB.
        """
        assert imgB.size(0) == points_congealed.size(0)
        N = imgB.size(0)
        num_points = points_congealed.size(1)
        source_res = imgB.size(-1) if input_img_for_sampling is None else input_img_for_sampling.size(-1)
        outB, gridB, flow_or_matrixB = self.forward(imgB, return_warp=True, return_flow=True,
                                                    output_resolution=output_resolution,
                                                    iters=iters, input_img_for_sampling=input_img_for_sampling,
                                                    **stn_forward_kwargs)

        # Compose the forward similarity transform (A --> congealed) and inverse similarity transform (congealed --> B)
        # This is the easier path since we there is a nice closed-form solution:
        if normalize_input_points:
            points_congealed = self.normalize(points_congealed, source_res, imgB.size(-1))
        if not self.is_flow:
            onehot = torch.tensor([[[0, 0, 1]]], dtype=torch.float, device=flow_or_matrixB.device).repeat(N, 1, 1)
            points_congealed = torch.cat([points_congealed, torch.ones(N, num_points, 1, device=points_congealed.device)], 2)  # (N, |points|, 3)
            congealed2B = torch.cat([flow_or_matrixB, onehot], 1).permute(0, 2, 1)
            pointsB = (points_congealed @ congealed2B)[..., [0, 1]]  # Apply transformation and remove homogeneous coordinate
        # Compose the forward flow (A --> congealed) and the inverse flow (congealed --> B)
        # While we have access to the inverse flows since our STN using reverse sampling, we do NOT have
        # access to the forward flows which complicates things. To approximate the A --> congealed forward flow,
        # we roughly reverse the congealed --> A inverse flow (which we have access to) via a brute-force nearest
        # neighbor search on the sampling grid.
        else:
            assert gridB.size(-1) == 2  # gridB is a flow field with shape (N, H, W, 2)
            pointsB = F.grid_sample(gridB.permute(0, 3, 1, 2), points_congealed.unsqueeze(2).float(), padding_mode='border').squeeze(3).permute(0, 2, 1)
        if unnormalize_output_points:
            pointsB = self.unnormalize(pointsB, imgB.size(-1), source_res)  # Back to coordinates in [0, H-1]
        return pointsB

    def transfer_points(self, imgA, imgB, pointsA, output_resolution=None, iters=1, **stn_forward_kwargs):
        """
        Given input images imgA, transfer known points pointsA to target images imgB.
        """
        assert imgA.size(0) == imgB.size(0) == pointsA.size(0)
        # Step 1: Map the key points in imgA to the congealed image (forward warp):
        points_congealed = self.congeal_points(imgA, pointsA, output_resolution=output_resolution,
                                           iters=iters, **stn_forward_kwargs)
        # Step 2: Map the key points in the congealed image to those in imgB (reverse warp):
        pointsB = self.uncongeal_points(imgB, points_congealed, output_resolution=output_resolution,
                                    normalize_input_points=False, iters=iters, **stn_forward_kwargs)
        return pointsB

    def load_state_dict(self, state_dict, strict=True):
        ignore = {'warp_head.one_hot', 'warp_head.rebias', 'input_downsample.kernel_horz',
                  'input_downsample.kernel_vert'}
        filtered = {k: v for k, v in state_dict.items() if k not in ignore}
        return super().load_state_dict(filtered, False)
