import torch
import torch.nn as nn
import math
from models.stylegan2.networks import EqualLinear, ConvLayer, ResBlock
from models.spatial_transformers.antialiased_sampling import BilinearDownsample


class ResnetClassifier(nn.Module):

    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], num_heads=1, supersize=None):
        super().__init__()
        self.stn_in_size = size
        self.num_heads = num_heads
        if supersize is not None:
            self.input_downsample = BilinearDownsample(supersize // size, 3)
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

        convs = [ConvLayer(3, int(channels[size]), 1)]
        log_size = int(math.log(size, 2))
        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(int(in_channel), int(out_channel), blur_kernel))
            in_channel = out_channel

        self.convs = nn.Sequential(*convs)
        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        self.to_logits = EqualLinear(channels[4] * 4 * 4, num_heads, activation='fused_lrelu')

    def forward(self, input):
        if input.size(-1) > self.stn_in_size:
            input = self.input_downsample(input)
        out = self.convs(input)
        out = self.final_conv(out)
        out = out.view(out.size(0), -1)
        out = self.to_logits(out)
        return out

    def assign(self, input, ignore_flips=False):
        preds = self.forward(input)
        classes = preds.argmax(dim=1)
        if ignore_flips:
            classes = classes % (self.num_heads // 2)
        return classes

    def run(self, input, target_cluster, return_flip_indices=False):
        num_nonflip_heads = self.num_heads // 2
        preds = self.forward(input)
        classes = preds.argmax(dim=1)
        keep_ixs, = torch.where((classes % num_nonflip_heads) == target_cluster)
        input = input[keep_ixs]
        flip_ixs = (classes[keep_ixs] >= num_nonflip_heads).reshape(keep_ixs.size(0), 1, 1, 1)
        input = torch.where(flip_ixs, input.flip(3,), input)
        if return_flip_indices:
            return input, preds[keep_ixs], flip_ixs, keep_ixs
        else:
            return input, preds[keep_ixs]

    def run_flip(self, input):
        num_nonflip_heads = self.num_heads // 2
        preds = self.forward(input)
        classes = preds.argmax(dim=1)
        flip_ixs = classes >= num_nonflip_heads
        input = torch.where(flip_ixs.reshape(input.size(0), 1, 1, 1), input.flip(3, ), input)
        return input, preds, classes, flip_ixs

    def run_flip_target(self, input, target_cluster):
        num_nonflip_heads = self.num_heads // 2
        preds = self.forward(input)[:, [target_cluster, target_cluster + num_nonflip_heads]]
        classes = preds.argmax(dim=1)
        flip_ixs = classes == 1
        input = torch.where(flip_ixs.reshape(input.size(0), 1, 1, 1), input.flip(3, ), input)
        return input, flip_ixs

    def run_flip_cartesian(self, input):
        num_nonflip_heads = self.num_heads // 2
        N = input.size(0)
        preds = self.forward(input)
        classes = preds.view(input.size(0), 2, num_nonflip_heads).argmax(dim=1)
        flip_ixs = classes == 1
        input = input.unsqueeze(1).repeat(1, num_nonflip_heads, 1, 1, 1)
        input = torch.where(flip_ixs.reshape(N, num_nonflip_heads, 1, 1, 1), input.flip(4, ), input)
        input = input.view(N * num_nonflip_heads, input.size(2), input.size(3), input.size(4))
        warp_policy = torch.eye(num_nonflip_heads, device=input.device).repeat(N, 1)
        return input, warp_policy

    def load_state_dict(self, state_dict, strict=True):
        ignore = {'input_downsample.kernel_horz', 'input_downsample.kernel_vert'}
        filtered = {k: v for k, v in state_dict.items() if k not in ignore}
        return super().load_state_dict(filtered, False)
