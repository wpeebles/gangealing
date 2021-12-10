/*
 * File   : splat_gpu_impl.cu
 * Author : Bill Peebles
 * Email  : peebles@berkeley.edu
 *
 */

#include "splat_gpu_impl.cuh"
#include <cstdio>
#include <cfloat>
#define _USE_MATH_DEFINES
#include <cmath>

#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)

#define CUDA_POST_KERNEL_CHECK \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (cudaSuccess != err) { \
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err)); \
            exit(-1); \
        } \
    } while(0)

#define CUDA_NUM_THREADS 32  // TODO: probably suboptimal

namespace {

static int CUDA_NUM_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__device__ static float GaussianPDF(const float mu_1, const float mu_2, const float x_1, const float x_2, const float normalizer)
{
    return exp(normalizer * (pow(x_1 - mu_1, 2.0f) + pow(x_2 - mu_2, 2.0f)));
}

__global__ void SplatForward(
        const int nthreads,
        F_DEVPTR_IN bottom_coordinates,
        F_DEVPTR_IN bottom_values,
        F_DEVPTR_IN bottom_sigma,
        F_DEVPTR_OUT top_alpha_splats,
        F_DEVPTR_OUT top_output,
        const int num_points,
        const int channels,
        const int height,
        const int width) {

  CUDA_KERNEL_LOOP(index, nthreads) {  // this is a loop over batch and point in coordinates
    // coordinates: (n, num_points, 2), (x,y)-coordinates
    // values: (n, num_points, channels)
    // (N, C, H, W) is an element in the output
    // int pw = index % width;
    // int ph = (index / width) % height;
    // int c = (index / width / height) % channels;
    const int n = index / num_points;
    const int pt = index % num_points;

    const float *this_coordinates = bottom_coordinates + 2 * num_points * n + 2 * pt;
    const float *this_value = bottom_values + channels * num_points * n + channels * pt;
    const float *this_sigma = bottom_sigma + n;
    float *this_alpha_splats = top_alpha_splats + n * height * width;
    float *this_out = top_output + n * channels * height * width;

    const float stdev = this_sigma[0];
    const float length = 2 * stdev;
    const float x_coord = this_coordinates[0];
    const float y_coord = this_coordinates[1];
    const float normalizer = -pow(2 * stdev * stdev, -1.0f);

    // Ignore out-of-bounds points:
    if ((x_coord >= 0 && x_coord < width) && (y_coord >= 0 && y_coord < height)) {

        const int t = fmax(0, floorf(y_coord - length));
        const int b = fmin(height - 1, ceilf(y_coord + length));
        const int l = fmax(0, floorf(x_coord - length));
        const int r = fmin(width - 1, ceilf(x_coord + length));

        for (int lh = t; lh <= b; ++lh) {
            for (int lw = l; lw <= r; ++lw) {
                // float alpha = pdf_alphas[lh * rect_width + lw] / alpha_total;
                float alpha = GaussianPDF(x_coord, y_coord, float(lw), float(lh), normalizer);
                atomicAdd(this_alpha_splats + lh * width + lw, alpha);
                // atomicAdd(this_alpha_splats + lh * width + lw, 1.0);
                for (int c = 0; c < channels; ++c) {
                    atomicAdd(this_out + c * height * width + lh * width + lw, alpha * this_value[c]);
                }
            }
        }
    }
  }
}

} /* !anonymous namespace */

#ifdef __cplusplus
extern "C" {
#endif

void SplatForwardGpu(
    cudaStream_t stream,
    F_DEVPTR_IN bottom_coordinates,
    F_DEVPTR_IN bottom_values,
    F_DEVPTR_IN bottom_sigma,
    F_DEVPTR_OUT top_alpha_splats,
    F_DEVPTR_OUT top_output,
    const int num_points_,
    const int channels_,
    const int height_,
    const int width_,
    const int top_count) {

    SplatForward<<<CUDA_NUM_BLOCKS(top_count), CUDA_NUM_THREADS, 0, stream>>>(
        top_count, bottom_coordinates, bottom_values, bottom_sigma, top_alpha_splats, top_output,
        num_points_, channels_, height_, width_);

    CUDA_POST_KERNEL_CHECK;
}

} /* !extern "C" */

