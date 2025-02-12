
#ifndef SR_CUDA_H
#define SR_CUDA_H

#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudaimgproc.hpp>
    
#ifdef __cplusplus
extern "C++" {
#endif

template <int scale_factor, int lanczos_a, int padding>
void cuda_lanczos_rs( const cv::cuda::GpuMat& src
    , cv::cuda::GpuMat& dst
    , cudaStream_t stream
    ) ;

void cuda_edge_aware_rs( const cv::cuda::GpuMat& src
                       , cv::cuda::GpuMat& dst
                       , int scale_factor
                       , cudaStream_t stream
                       ) ;

void cuda_wnnm_rs( const cv::cuda::GpuMat& src
                 , cv::cuda::GpuMat& dst
                 , int scale_factor
                 , cudaStream_t stream
                 ) ;
                 
#ifdef __cplusplus
}
#endif

#endif 