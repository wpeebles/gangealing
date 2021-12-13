"""Spatial transformer with mipmap-based anti-aliasing."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Warp(nn.Module):
    """Module for applying spatial transforms without anti-aliasing."""

    def __init__(self):
        super(Warp, self).__init__()

    def forward(self, inputs: torch.Tensor, grid: torch.Tensor, padding_mode: str = 'border') -> torch.Tensor:
        return F.grid_sample(inputs, grid, padding_mode=padding_mode, align_corners=False)


class MipmapWarp(nn.Module):
    """
    Module for applying spatial transforms with mipmap anti-aliasing.
    Code from Tim Brooks.
    """

    def __init__(self, max_num_levels: int = 8):
        """Initializes MipmapWarp class.
        Args:
          max_num_levels (int, optional): Max number of mipmap levels (default 8).
        """
        super(MipmapWarp, self).__init__()
        self.max_num_levels = max_num_levels
        self._register_blur_filter()
        self.levels_map = None

    def forward(  # pylint: disable=arguments-differ
            self, inputs: torch.Tensor, grid: torch.Tensor,
            min_level: float = 0.0, padding_mode: str = 'border') -> torch.Tensor:
        """Applies spatial transform with antialiasing; analogous to grid_sample().
        Args:
          inputs (torch.Tensor): Input features on which to apply transform.
          grid (torch.Tensor): Sampling grid normalized to [-1, 1].
        Returns:
            torch.Tensor: Transformed features.
        """
        # Determines level in mipmap stack to sample at each pixel.
        _, _, height, width = inputs.size()
        coords = self._get_coordinates(grid, height, width)
        levels = self._get_mipmap_levels(coords, self.max_num_levels)
        levels = levels.clamp(min=min_level)

        # Computes total number of levels needed in stack for this sampling.
        num_levels = int(levels.max().ceil().item()) + 1

        # Creates a stack of Gaussian filtered features and warps each level.
        stack = self._create_stack(inputs, num_levels)
        stack = self._warp_stack(stack, grid, padding_mode=padding_mode)

        outputs = self._sample_mipmap(stack, levels)
        self.levels_map = levels / (self.max_num_levels - 1.0)
        return outputs

    @staticmethod
    def get_max_coord_distance(coords: torch.Tensor) -> torch.Tensor:
        """Computes max distance of neighboring coordinates.
        Args:
          coords (torch.Tensor): Coordinates of shape [N, H, W, 2].
        Returns:
          torch.Tensor: Maximum distances of shape [N, H, W].
        """
        # pylint: disable=too-many-locals

        # Pads coordinates.
        coords_padded = coords.permute(0, 3, 1, 2)
        coords_padded = nn.ReplicationPad2d(1)(coords_padded)
        coords_padded = coords_padded.permute(0, 2, 3, 1)
        # Gets neighboring coordinates on four sides of each sample.
        coords_l = coords_padded[:, 1:-1, :-2, :]
        coords_r = coords_padded[:, 1:-1, 2:, :]
        coords_u = coords_padded[:, :-2, 1:-1, :]
        coords_d = coords_padded[:, 2:, 1:-1, :]

        # Computes distance between coordinates and each neighbor.
        def _get_dist(other_coords):
            sq_dist = torch.sum((other_coords - coords) ** 2, dim=3)
            # Clamps at 1 to prevent numerical instability of square root. Does not
            # introduce bias since log2(1) = 0, which is the lowest mipmap level.
            return sq_dist.clamp(min=1.0) ** 0.5

        dist_l = _get_dist(coords_l)
        dist_r = _get_dist(coords_r)
        dist_u = _get_dist(coords_u)
        dist_d = _get_dist(coords_d)
        dists = torch.stack([dist_l, dist_r, dist_u, dist_d])

        # Determines stack level from maximum distance at each sample.
        dist_max, _ = torch.max(dists, dim=0)
        return dist_max

    ##############################################################################
    # Private Instance Methods
    ##############################################################################

    def _register_blur_filter(self):
        """Registers a Gaussian blurring filter to the module."""
        blur_filter = np.array([1., 3., 3., 1.])
        blur_filter = torch.Tensor(blur_filter[:, None] * blur_filter[None, :])
        blur_filter = blur_filter / torch.sum(blur_filter)
        blur_filter = blur_filter[None, None, ...]
        self.register_buffer('blur_filter', blur_filter)

    def _downsample_2x(self, inputs: torch.Tensor) -> torch.Tensor:
        """Gaussian blurs inputs along spatial dimensions."""
        num_channels = inputs.shape[1]
        blur_filter = self.blur_filter.repeat((num_channels, 1, 1, 1))
        inputs = nn.ReflectionPad2d(1)(inputs)
        outputs = F.conv2d(inputs, blur_filter, stride=2, groups=num_channels)
        return outputs

    def _create_stack(self, inputs: torch.Tensor,
                      num_levels: int) -> torch.Tensor:
        """Creates a Gaussian stack; blurs each level, but does not downsample.
        Args:
          inputs (torch.Tensor): Input features of shape [N, C, H, W].
          num_levels (int): Number of levels to create in stack.
        Returns:
          torch.Tensor: Gaussian stack of shape [N, C, D, H, W], were dimension
              D represents stack level.
        """
        # _, _, height, width = inputs.size()
        log_size = np.log2(inputs.size(-1))
        pad_needed = not log_size.is_integer()
        if pad_needed:
            target_size = 2 ** np.ceil(log_size)
            total_pad = target_size - inputs.size(-1)
            left_pad = int(total_pad // 2)
            right_pad = int(total_pad - left_pad)
            inputs = F.pad(inputs, pad=(left_pad, right_pad, left_pad, right_pad), mode='reflect')
        levels = [inputs]

        for i in range(1, num_levels):
            inputs = self._downsample_2x(inputs)
            scale_factor = 2.0 ** i
            level = self._upsample(inputs, scale_factor)
            levels.append(level)

        stack = torch.stack(levels, dim=2)
        if pad_needed:
            stack = stack[:, :, :, left_pad:-right_pad, left_pad:-right_pad]
        return stack

    ##############################################################################
    # Private Static Methods
    ##############################################################################

    @staticmethod
    def _upsample(inputs: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """Bilinear upsampling."""
        outputs = F.interpolate(
            inputs, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        return outputs

    @staticmethod
    def _warp_stack(stack: torch.Tensor, grid: torch.Tensor,
                    padding_mode='border') -> torch.Tensor:
        """Applies F.grid_sample() to each level of a stack.
        Args:
          stack (torch.Tensor): Stack of shape [N, C, D, H_in, W_in].
          grid (torch.Tensor): Sampling grid of shape [N, H_out, W_out, 2].
          padding_mode: 'zeros' or 'border' or 'reflection'.
        Returns:
          torch.Tensor: Warped stack of shape [N, C, D, H_out, W_out].
        """
        N, C, D, H_in, W_in = stack.size()  # pylint: disable=invalid-name
        _, H_out, W_out, _ = grid.size()  # pylint: disable=invalid-name

        stack = stack.reshape((N, C * D, H_in, W_in))
        stack = F.grid_sample(stack, grid, padding_mode=padding_mode, align_corners=False)
        stack = stack.reshape((N, C, D, H_out, W_out))
        return stack

    @staticmethod
    def _get_coordinates(grid: torch.Tensor, height: int,
                         width: int) -> torch.Tensor:
        """Converts a normalized grid in [-1, 1] to absolute coordinates.
        Args:
          grid (torch.Tensor): Sampling grid of shape [N, H, W, 2].
          height (int): Height of the source being sampled.
          width (int): Width of the source being sampled.
        Returns:
          torch.Tensor: Coordinates of shape [N, H, W, 2].
        """
        x_coord = (width - 1.0) * (grid[..., 0] + 1.0) / 2.0
        y_coord = (height - 1.0) * (grid[..., 1] + 1.0) / 2.0
        coords = torch.stack([x_coord, y_coord], dim=3)
        return coords

    @staticmethod
    def _get_mipmap_levels(coords: torch.Tensor,
                           max_num_levels: int) -> torch.Tensor:
        """Computes level in mipmap to sample at each pixel based on coordinates.
        Args:
          coords (torch.Tensor): Coordinates of shape [N, H, W, 2].
          max_num_levels (int): Max number of levels allowed in mipmap.
        Returns:
          torch.Tensor: Mipmap levels of shape [N, H, W].
        """
        dist_max = MipmapWarp.get_max_coord_distance(coords)
        levels = torch.log2(dist_max)
        levels = levels.clamp(min=0.0, max=max_num_levels - 1.0)
        return levels

    @staticmethod
    def _sample_mipmap(stack: torch.Tensor,
                       levels: torch.Tensor) -> torch.Tensor:
        """Linearly samples mipmap stack at levels.
        Args:
          stack (torch.Tensor): Gaussian stack of shape [N, C, D, H, W].
          levels (torch.Tensor): Mipmap levels of shape [N, H, W].
        Returns:
          torch.Tensor: Output samples of shape [N, C, H, W].
        """
        # Adds channel dim of size C and level dim of size 1.
        C = stack.shape[1]  # pylint: disable=invalid-name
        levels = torch.stack([levels] * C, dim=1)
        levels = levels[:, :, None, :, :]

        # Gets two levels to interpolate between at each pixel.
        level_0 = levels.floor().long()
        level_1 = levels.ceil().long()
        level_dim = 2
        output_0 = torch.gather(stack, level_dim, level_0)
        output_1 = torch.gather(stack, level_dim, level_1)

        # Linearly interpolates between levels.
        weight = levels % 1.0
        output = output_0 + weight * (output_1 - output_0)
        output = output[:, :, 0, :, :]
        return output


class BilinearDownsample(nn.Module):  # From Richard Zhang
    def __init__(self, stride, channels):
        super().__init__()
        self.stride = stride
        self.channels = channels
        # create tent kernel
        kernel = np.arange(1, 2 * stride + 1, 2)  # ramp up
        kernel = np.concatenate((kernel,kernel[::-1]))  # reflect it and concatenate
        kernel = torch.Tensor(kernel/np.sum(kernel))  # normalize
        self.register_buffer('kernel_horz', kernel[None, None, None, :].repeat((self.channels, 1, 1, 1)))
        self.register_buffer('kernel_vert', kernel[None, None, :, None].repeat((self.channels, 1, 1, 1)))
        self.refl = nn.ReflectionPad2d(int(stride/2))

    def forward(self, input):
        return F.conv2d(F.conv2d(self.refl(input), self.kernel_horz, stride=(1, self.stride), groups=self.channels),
                        self.kernel_vert, stride=(self.stride, 1), groups=self.channels)


if __name__ == '__main__':
    warp = MipmapWarp(3.5)
    x = torch.randn(1, 3, 450, 450)
    grid = F.affine_grid(torch.eye(2, 3).unsqueeze_(0), (1, 3, 128, 128))
    print(warp(x, grid).size())
