



#ifndef DENOISE_CUDA_H
#define DENOISE_CUDA_H

#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudaimgproc.hpp>
    
#ifdef __cplusplus
extern "C++" {
#endif

template <int RADIUS>
void cuda_bilateral_filter( const cv::cuda::GpuMat& src
                          , cv::cuda::GpuMat& dst
                          , float sigma_color
                          , float sigma_space
                          , bool use_lab
                          , cudaStream_t stream = nullptr
                          ) ;

template <int PATCH_SIZE, int SEARCH_WINDOW>
void cuda_nlmeans_filter( const cv::cuda::GpuMat& src
                        , cv::cuda::GpuMat& dst
                        , float h = 0.4f
                        , cudaStream_t stream = nullptr
                        ) ;

template <int RADIUS>
void cuda_median_filter( const cv::cuda::GpuMat& src
                       , cv::cuda::GpuMat& dst
                       , cudaStream_t stream = nullptr
                       ) ;


#ifdef __cplusplus
}
#endif

#endif