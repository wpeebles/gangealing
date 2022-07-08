import torch.autograd as ag
from .splat_cpu import splat_forward_cpu
__all__ = ['splat2d']


_splat = None


class Splat2DFunction(ag.Function):

    @staticmethod
    def forward(ctx, input, coordinates, values, sigma, soft_normalize=False):

        assert 'FloatTensor' in coordinates.type() and 'FloatTensor' in values.type(), \
                'Splat2D only takes float coordinates and values, got {} and {} instead.'.format(coordinates.type(), values.type())
        assert coordinates.size(0) == values.size(0) and coordinates.size(1) == values.size(1), \
            'coordinates should be size (N, num_points, 2) and values should be size (N, num_points, *), got {} and {} instead.'.format(coordinates.shape, values.shape)
        assert input.size(0) == coordinates.size(0) and input.dim() == 4, 'input should be of size (N, *, H, W), got {} instead'.format(input.shape)
        assert sigma.size(0) == input.size(0), 'sigma should be a tensor of size (N,)'

        input = input.contiguous()
        coordinates = coordinates.contiguous()
        values = values.contiguous()
        sigma = sigma.contiguous()

        # Apply splatting
        output = splat_forward_cpu(input, coordinates, values, sigma, soft_normalize)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


splat2d = Splat2DFunction.apply
