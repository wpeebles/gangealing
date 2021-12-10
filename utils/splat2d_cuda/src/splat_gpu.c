#include <math.h>
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>

#include "splat_gpu_impl.cuh"


at::Tensor splat_forward_cuda(const at::Tensor &input, const at::Tensor &coordinates, const at::Tensor &values,
                              const at::Tensor &sigma, const bool soft_normalize) {
    int nr_imgs = input.size(0);
    int nr_points = coordinates.size(1);
    int nr_channels = input.size(1);
    int top_count = nr_imgs * nr_points;
    int height = input.size(2);
    int width = input.size(3);
    auto alpha_splats = at::zeros({nr_imgs, height, width}, values.options());
    auto output = at::clone(input);

    if (output.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return output;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    SplatForwardGpu(
        stream, coordinates.data_ptr<float>(), values.data_ptr<float>(), sigma.data_ptr<float>(),
        alpha_splats.data_ptr<float>(), output.data_ptr<float>(),
        nr_points, nr_channels, height, width, top_count);

    THCudaCheck(cudaGetLastError());

    alpha_splats = alpha_splats.view({nr_imgs, 1, height, width});
    if (soft_normalize) {
        alpha_splats = alpha_splats.clamp(1.0f);
    }
    output = output / (alpha_splats + 1e-8);
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("splat_forward_cuda", &splat_forward_cuda, "Splat_forward");
}
