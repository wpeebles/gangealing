#ifndef SPLAT_GPU_IMPL_CUH
#define SPLAT_GPU_IMPL_CUH

#ifdef __cplusplus
extern "C" {
#endif

#define F_DEVPTR_IN const float *
#define F_DEVPTR_OUT float *

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
    const int top_count);

#ifdef __cplusplus
} /* !extern "C" */
#endif

#endif /* !SPLAT_GPU_IMPL_CUH */

