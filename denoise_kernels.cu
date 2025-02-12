


#include "denoise_cuda.h"

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_types.hpp"
#include <opencv2/cudaarithm.hpp>
#include "opencv2/cudaimgproc.hpp"
#include <algorithm>
#include <device_launch_parameters.h>
#include <opencv2/core/cuda_stream_accessor.hpp>

__device__ __forceinline__
float color_distance_lab(uchar3 p1, uchar3 p2) {
    float dl = p1.x - p2.x;
    float da = p1.y - p2.y;
    float db = p1.z - p2.z;
    return dl * dl + da * da + db * db;
}

__device__ __forceinline__
float color_distance_RGB(uchar3 p1, uchar3 p2) {
    float dr = p1.x - p2.x;
    float dg = p1.y - p2.y;
    float db = p1.z - p2.z;
    return dr * dr + dg * dg + db * db;
}

template <int RADIUS>
__global__ void bilateral_kernel( cv::cuda::PtrStepSz<uchar3> src
                                , cv::cuda::PtrStepSz<uchar3> dst
                                , const float sigma_color
                                , const float sigma_space
                                , const bool use_lab
                                ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= src.cols || y >= src.rows) return;

    constexpr int PAD = RADIUS;
    constexpr int SHARED_WIDTH = 16 + 2 * PAD;
    constexpr int SHARED_HEIGHT = 16 + 2 * PAD;
    __shared__ uchar3 shared_block[SHARED_HEIGHT][SHARED_WIDTH];

    const int lx = threadIdx.x;
    const int ly = threadIdx.y;

    shared_block[ly + PAD][lx + PAD] = src(y, x);

    for (int offset = 0; offset < (2 * PAD + 1) * (2 * PAD + 1); offset += blockDim.x * blockDim.y) {
        int idx = offset + threadIdx.y * blockDim.x + threadIdx.x;
        int di = idx / (2 * PAD + 1) - PAD;
        int dj = idx % (2 * PAD + 1) - PAD;

        if (di == 0 && dj == 0) continue;

        int load_x = x + dj;
        int load_y = y + di;
        load_x = max(0, min(load_x, src.cols - 1));
        load_y = max(0, min(load_y, src.rows - 1));

        int shared_row = ly + di + PAD;
        int shared_col = lx + dj + PAD;

        if (shared_row >= 0 && 
            shared_row < SHARED_HEIGHT && 
            shared_col >= 0 && 
            shared_col < SHARED_WIDTH) shared_block[shared_row][shared_col] = src(load_y, load_x);
    }
    __syncthreads();

    const uchar3 center = shared_block[ly + PAD][lx + PAD];
    float3 sum = {0.f, 0.f, 0.f};
    float total_weight = 0.f;

    for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
            const int sy = ly + PAD + dy;
            const int sx = lx + PAD +dx;

            if ( sy < 0 || 
                 sy >= blockDim.y + 2 * PAD || 
                 sx < 0 || 
                 sx >= blockDim.x + 2 * PAD ) continue;
            
            const uchar3 sample = shared_block[sy][sx];

            const float space_dist = (dx * dx + dy * dy) / (sigma_space * sigma_space);
            
            float color_dist;
            if (use_lab) color_dist = color_distance_lab(center, sample) / (sigma_color * sigma_color);
            else         color_dist = color_distance_RGB(center, sample) / (sigma_color * sigma_color);

            const float weight = __expf(-(space_dist + color_dist));

            sum.x += sample.x * weight;
            sum.y += sample.y * weight;
            sum.z += sample.z * weight;
            total_weight += weight;
        }
    }

    if (total_weight > 1e-6f) {
        dst(y, x) = make_uchar3(__float2uint_rn( sum.x /total_weight)
                                                  , __float2uint_rn(sum.y / total_weight)
                                                  , __float2uint_rn(sum.z / total_weight)
                                                  ) ;
    } else dst(y, x) = center;
}

template <int RADIUS>
void cuda_bilateral_filter( const cv::cuda::GpuMat& src
                          , cv::cuda::GpuMat& dst
                          , float sigma_color
                          , float sigma_space
                          , bool use_lab
                          , cudaStream_t stream
                          ) {
    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);
    cv::cuda::GpuMat lab_src;

    if (use_lab) {
        cv::cuda::cvtColor(src, lab_src, cv::COLOR_BGR2Lab, 0, cv_stream);
    }

    constexpr int block_size = 16;
    dim3 grid((src.cols + block_size - 1) / block_size, (src.rows + block_size) / block_size); 
    dim3 block(block_size, block_size);

    bilateral_kernel<RADIUS><<<grid, block, 0, stream>>>(
        use_lab ? lab_src : src,
        dst,
        sigma_color,
        sigma_space,
        use_lab
    );

}

template void cuda_bilateral_filter<4>( const cv::cuda::GpuMat&
                                      , cv::cuda::GpuMat&
                                      , float
                                      , float
                                      , bool
                                      , cudaStream_t
                                      ) ;


#include <math.h>

template <int PATCH_SIZE, int SEARCH_WINDOW>
__device__ __forceinline__
float compute_similarity( const uchar* __restrict__ src
                        , int x1, int y1
                        , int x2, int y2
                        , int rows, int cols
                        , int step) {
    float sum_sqdiff = 0.f;
    for (int dy = -PATCH_SIZE; dy <= PATCH_SIZE; ++dy) {
        for (int dx = -PATCH_SIZE; dx <= PATCH_SIZE; ++dx) {
            int yy1 = y1 + dy;
            int xx1 = ::max(0, ::min(x1 + dx, cols - 1));
            int yy2 = y2 + dy;
            int xx2 = ::max(0, ::min(x2 + dx, cols - 1));

            yy1 = ::max(0, ::min(yy1, rows - 1));
            yy2 = ::max(0, ::min(yy2, rows - 1));
            float diff = src[yy1 * step + xx1] - src[yy2 * step + xx2];
            sum_sqdiff += diff * diff;
        }
    }
    return sum_sqdiff;
}

template <int PATCH_SIZE, int SEARCH_WINDOW>
__global__ void nlmeans_kernel( cv::cuda::PtrStepSz<uchar> src
                              , cv::cuda::PtrStepSz<uchar> dst
                              , float h
                              , float border
                              ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x + border;
    const int y = blockIdx.y * blockDim.y + threadIdx.y + border;

    if (x >= src.cols - border || y >= src.rows - border) return;

    constexpr int PATCH_RADIUS = PATCH_SIZE;
    constexpr int SEARCH_RADIUS = SEARCH_WINDOW / 2;

    __shared__ uchar shared_memory[16 + 2 * SEARCH_RADIUS][16 + 2 * SEARCH_RADIUS];

    for (int i = -SEARCH_RADIUS; i < SEARCH_RADIUS + blockDim.y; ++i) {
        for (int j = -SEARCH_RADIUS; j < SEARCH_RADIUS + blockDim.x; ++j) {
            int yy = ::max(0, ::min(y + i - static_cast<int>(blockDim.y / 2), src.rows - 1));
            int xx = ::max(0, ::min(x + j - static_cast<int>(blockDim.x / 2), src.cols - 1));
            shared_memory[threadIdx.y + i + SEARCH_RADIUS][threadIdx.x + j + SEARCH_RADIUS] = src(yy, xx);
        }
    }
    __syncthreads();

    float sum_weights = 0.f;
    float sum_pixels = 0.f;
    const uchar center = src(y, x);

    for (int dy = -SEARCH_RADIUS; dy <= SEARCH_RADIUS; ++dy) {
        for (int dx = -SEARCH_RADIUS; dx <= SEARCH_RADIUS; ++dx) {
            int yy = y + dy;
            int xx = x + dx;
            if ( xx >= border &&
                 x < src.cols - border &&
                 y >= border &&
                 yy < src.rows - border
               ) {
                float distance = compute_similarity<PATCH_SIZE, SEARCH_WINDOW>( src.data
                                                                              , x, y
                                                                              , xx, yy
                                                                              , src.rows, src.cols
                                                                              , src.step
                                                                              ) ;
                float weight = expf(-distance / (h * h));
                sum_weights += weight;
                sum_pixels += shared_memory[threadIdx.y + dy + SEARCH_RADIUS][threadIdx.x + dx + SEARCH_RADIUS] * weight;
            }
        }
    }

    if(sum_weights > 1e-6f) dst(y, x) = __float2uint_rd(sum_pixels /  sum_weights);
    else dst(y, x)  = center;
}

template <int PATCH_SIZE, int SEARCH_WINDOW>
void cuda_nlmeans_filter( const cv::cuda::GpuMat& src
                        , cv::cuda::GpuMat& dst
                        , float h
                        , cudaStream_t stream
                        ) {
    
    const int border = PATCH_SIZE + (SEARCH_WINDOW / 2);
    dst.create(src.size(), src.type());

    dim3 block(16, 16);
    dim3 grid( (src.cols + block.x - 1) / block.x
             , (src.rows + block.y - 1) / block.y
             ) ;

    if (src.channels() == 1) nlmeans_kernel<PATCH_SIZE, SEARCH_WINDOW><<<grid, block, 0, stream>>>(src, dst, h, border);
    else {
        std::vector<cv::cuda::GpuMat> channels;
        cv::cuda::split(src, channels);
        
        for (auto& ch: channels) 
            nlmeans_kernel<PATCH_SIZE, SEARCH_WINDOW><<<grid, block, 0, stream>>>(ch, ch, h, border);
        cv::cuda::merge(channels, dst);
    }
    cudaStreamSynchronize(stream);
}

template void cuda_nlmeans_filter<5, 21>(const cv::cuda::GpuMat&, cv::cuda::GpuMat&, float, cudaStream_t);














