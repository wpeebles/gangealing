"""
Pytorch-based implementation of Laplacian Pyramid Blending (supports batching, GPU and CPU).
Author: Bill Peebles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cv2 import getGaussianKernel
from math import sqrt


class LaplacianBlender(nn.Module):

    """
    Differentiable implementation of Laplacian Pyramid Blending with GPU and batching support.
    """

    def __init__(self, levels=5, gaussian_kernel_size=45, gaussian_sigma=1,
                 level_size_adder=0, level_sigma_multiplier=2):
        """
        :param levels: Number of levels in the pyramid (default=10). More levels will take longer.
        :param gaussian_kernel_size: Width of Gaussian filter used for blurring the images.
        :param gaussian_sigma: Standard deviation of the Gaussian filter the images.
        :param level_size_adder: Amount by which to increase the Gaussian kernel size when moving down the pyramid.
        :param level_sigma_multiplier: Factor by which to increase the Gaussian sigma when moving down the pyramid.
        """
        super().__init__()
        assert gaussian_kernel_size % 2 == 1, 'gaussian_kernel_size needs to be odd for easier padding'
        assert level_size_adder % 2 == 0, 'level_size_adder needs to be even for easier padding'

        self.levels = levels
        self.kernel_padding = []
        for level in range(self.levels):
            width = gaussian_kernel_size + level_size_adder
            sigma = gaussian_sigma * level_sigma_multiplier ** level
            kernel = self.gauss2d(width, sigma)
            self.register_buffer(f'kernel_{level}', kernel)
            self.kernel_padding.append(width // 2)

    @staticmethod
    def gauss2d(ksize, sigma, num_channels=3):
        """
        Constructs a 2D Gaussian filter for blurring.

        :param ksize: Width of the filter
        :param sigma: Standard deviation of the Gaussian
        :param num_channels: Number of channels in the input that this filter will be applied to
        :return: (num_channels, 1, ksize, ksize) 2D Gaussian filter for use with torch.nn.functional.conv2d
        """
        gauss_filter_1d = getGaussianKernel(ksize, sigma).reshape((ksize, 1))
        gauss_filter_2d = gauss_filter_1d @ gauss_filter_1d.T
        out = torch.from_numpy(gauss_filter_2d).view(1, 1, ksize, ksize).repeat(num_channels, 1, 1, 1).float()
        return out

    def get_stacks(self, img):
        """
        Constructs Laplacian stacks and Gaussian stacks.

        :param img: (N, C, H, W) image tensor
        :return: Two (self.levels, N, C, H, W) tensors: the Laplacian and Gaussian stacks for img
        """
        lap_stack = []
        gauss_stack = []
        num_channels = img.size(1)
        for level in range(self.levels):
            gauss_stack.append(img)
            if level < self.levels - 1:
                g = getattr(self, f'kernel_{level}')[:num_channels]
                padding = self.kernel_padding[level]  # SAME padding
                img_pad = F.pad(img, (padding, padding, padding, padding), mode='replicate')
                blur_img = F.conv2d(img_pad, g, groups=num_channels)
                lap_stack.append(img - blur_img)
                img = blur_img
            else:
                lap_stack.append(img)
        return torch.stack(lap_stack), torch.stack(gauss_stack)

    def forward(self, img0, img1, mask):
        """
        Blends img0 and img1 according to mask. In principle, img0 and img1 can have any dynamic range
        (e.g., -1 to +1 or 0 to 255), but ranges like 0 to 255 might be more prone to overflow. So using a normalized
        dynamic range might be more stable. mask should always have a dynamic range of 0 to 1.

        :param img0: (N, C, H, W) image tensor
        :param img1: (N, C, H, W) image tensor
        :param mask: (N, 1, H, W) tensor with values between 0 and 1 (inclusive). Where mask == 0,
               pixels from img0 will be fully taken. Where mask == 1, pixels from img1 will be fully taken.
        :return: (N, C, H, W) image tensor: the result of Laplacian Blending img0 and img1 according to mask.
        """

        # Make sure inputs have correct dimensions and consistent sizes:
        assert img0.dim() == img1.dim() == mask.dim(), \
            'LaplacianBlender.forward expects (N, C, H, W) input tensors'
        assert mask.size(1) == 1, f'mask input should have num_channels==1, but got num_channels=={mask.size(1)}'
        assert img0.size() == img1.size(), \
            f'img0 and img1 should be the same size but got img0 size of {img0.size()} and img1 size of {img1.size()}'
        assert mask.size()[2:] == img0.size()[2:], \
            f'mask should have the same batch size and spatial resolution as img0 and img1, ' \
            f'but got mask size of {mask.size()}'

        lp0, _ = self.get_stacks(img0)  # Laplacian stack for img0
        lp1, _ = self.get_stacks(img1)  # Laplacian stack for img1
        _, gpm = self.get_stacks(mask)  # Gaussian stack for mask
        blended_stack = lp0.lerp(lp1, gpm)  # Apply alpha compositing to the two Laplacian stacks via the blurred masks
        blended_image = blended_stack.sum(dim=0)  # Collapse the stack to get the final blended image
        return blended_image


def extend_object_border(img, mask, max_pixel_radius=45):
    """
    This function works nicely with Laplacian Pyramid Blending to "extend" the borders of an object. You can think of
    this function as grabbing the object, moving it around in circles of increasing radius, and "stamping" the object
    pixels as it is moved around in a circle. Think of it as a type of object-centric border padding within an image.
    :param img: (N, C, H, W) image tensor, usually contains an image of a object (most pixels should be 0.0)
    :param mask: (N, 1, H, W) (soft-)mask tensor indicating where the object lies in img
    :param max_pixel_radius: int that indicating the radius of the final circle
    :return: (N, C, H, W) image tensor; the result of moving the object in img around in circles and pasting the result
    """
    assert img.dim() == mask.dim() == 4
    out = img.clone()
    mask = mask.clone()
    original_mask = mask.clone()
    for radius in range(1, max_pixel_radius + 1):
        sqrt_rad = round(radius / sqrt(2))
        # Coarsely sample 8 points on a circle's circumference:
        points = [(radius, 0), (-radius, 0), (0, radius), (0, -radius), (sqrt_rad, sqrt_rad),
                  (-sqrt_rad, sqrt_rad), (sqrt_rad, -sqrt_rad), (-sqrt_rad, -sqrt_rad)]
        for pv, ph in points:
            img_shifted = img.roll((pv, ph), dims=(2, 3))
            mask_shifted = original_mask.roll((pv, ph), dims=(2, 3))
            out.add_(img_shifted.mul_(1 - mask))
            mask.add_(mask_shifted.clone()).clamp_(min=0.0, max=1.0)
    return out


if __name__ == '__main__':
    # Simple example usage on GPU:
    device = 'cuda'  # also can run in cpu mode instead
    # x = torch.randn(5, 3, 144, 201, device=device)
    # y = torch.randn_like(x)
    # mask = torch.rand(5, 1, 144, 201, device=device)
    blender = LaplacianBlender(levels=5).to(device)
    # b = blender(x, y, mask)
    # print(b.size())

    from PIL import Image
    x0 = Image.open('fp1.png').convert('RGB')
    x1 = Image.open('fp0.png').convert('RGB')
    m = Image.open('fm.png')
    import numpy as np
    x0 = torch.from_numpy(np.asarray(x0)).float().permute(2, 0, 1).unsqueeze_(0).div(255.0).add(-0.5).mul(2.0).to(device)
    x1 = torch.from_numpy(np.asarray(x1)).float().permute(2, 0, 1).unsqueeze_(0).div(255.0).add(-0.5).mul(2.0).to(device)
    m = torch.from_numpy(np.asarray(m)).float().permute(2, 0, 1).unsqueeze_(0).div(255.0).to(device)[:, 0:1]
    x1 = x1 * m
    x1 = extend_object_border(x1, m)
    b = blender(x0, x1, m)
    from torchvision.utils import save_image
    # save_image(ext, 'e.png', normalize=True, range=(-1, 1))
    # save_image(hm, 'm.png', normalize=True, range=(0, 1), nrow=4)
    save_image(x1, 'e.png', normalize=True, range=(-1, 1))
    save_image(b, 'b.png', normalize=True, range=(-1, 1))
