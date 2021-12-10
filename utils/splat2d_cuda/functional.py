import torch.autograd as ag

__all__ = ['splat2d']


_splat = None


def _import_splat():
    global _splat

    if _splat is None:
        try:
            from os.path import join as pjoin, dirname
            from torch.utils.cpp_extension import load as load_extension
            root_dir = pjoin(dirname(__file__), 'src')

            _splat = load_extension(
                '_splat',
                [pjoin(root_dir, 'splat_gpu.c'), pjoin(root_dir, 'splat_gpu_impl.cu')],
                extra_cflags=['-O3'],
                extra_cuda_cflags=['-O3'],
                verbose=True
            )
        except ImportError:
            raise ImportError('Cannot compile splatting CUDA library.')

    return _splat


class Splat2DFunction(ag.Function):

    @staticmethod
    def forward(ctx, input, coordinates, values, sigma, soft_normalize=False):
        _splat = _import_splat()

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

        if coordinates.is_cuda:
            output = _splat.splat_forward_cuda(input, coordinates, values, sigma, soft_normalize)
            ctx.params = ()
            # everything here is contiguous.
            # ctx.save_for_backward(features, rois, output)
        else:
            raise NotImplementedError('Splat2D currently only has support for GPU (cuda).')

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


splat2d = Splat2DFunction.apply
