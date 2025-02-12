#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include "sr_cuda.h"

#include "opencv2/core.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/stitching/detail/warpers.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <opencv2/core/cuda/common.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cuda_texture_types.h>
#include <surface_functions.h>
#include <surface_indirect_functions.h>
#include <opencv2/core/cuda_stream_accessor.hpp>

__device__ __forceinline__
uchar saturate_uchar(float x) {
    return (uchar)(__saturatef(x) * 255.f);
}

#define PI 3.1415926

__device__ __forceinline__
float lanczos_window(float x, int a) {
    if (x == 0.f) return 1.f;
    if (x < -a || x >= a) return 0.f;
    const float pi_x = CUDART_PI_F * x;
    return (a * __sinf(pi_x) * __sinf(pi_x / a)) / (pi_x * pi_x);
}

template <int scale_factor, int lanczos_a = 4, int padding = 3>
__global__ void lanczos_upscale_kernel( cudaTextureObject_t tex_src, cudaSurfaceObject_t surf_dst
                                      , const int src_width, const int src_height
                                      ) {
    extern __shared__ float4 shared_tile[];
                                        
    const int tile_w = blockDim.x + 2 * padding;
    const int tile_h = blockDim.y + 2 * padding;

    for (int dy = -padding; dy < blockDim.y + padding; dy += blockDim.y) {
        for (int dx = -padding; dx < blockDim.x + padding; dx += blockDim.x) {
            const int src_x = blockIdx.x * blockDim.x + dx + threadIdx.x - padding;
            const int src_y = blockIdx.y * blockDim.y + dy + threadIdx.y - padding;

            int clamp_x = ::max(0, ::min(src_width - 1, src_x));
            int clamp_y= ::max(0, ::min(src_height - 1, src_y));

            float4 color = tex2D<float4>(tex_src, clamp_x + 0.5f, clamp_y + 0.5f);
            
            int s_idx = (threadIdx.y + dy + padding) * tile_w + threadIdx.x + dx + padding;
            if (s_idx < tile_w * tile_w) shared_tile[s_idx] = color;
        }
    }
    __syncthreads();

    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= src_width * scale_factor ||
        dst_y >= src_height * scale_factor ) return;

    const float src_x = (dst_y + 0.5f) / scale_factor;
    const float src_y = (dst_y + 0.5f) / scale_factor;

    float4 sum = {0.f, 0.f, 0.f, 0.f};
    float total_weight = 0.f;

    const int center_x = __float2int_rd(src_x);
    const int center_y = __float2int_rd(src_y);

    for (int dy = -lanczos_a + 1; dy <= lanczos_a; ++dy) {
        for (int dx = -lanczos_a + 1; dx <= lanczos_a; ++dx) {
            const int sample_x = center_x + dx;
            const int sample_y = center_y + dy;

            if (sample_x < 0 || sample_x >= src_width ||
                sample_y < 0 || sample_y >= src_height ) continue;

            const float dist_x = src_x - (sample_x + 0.5f);
            const float dist_y = src_y - (sample_y + 0.5f);

            const float w_x = lanczos_window(dist_x, lanczos_a);
            const float w_y = lanczos_window(dist_y, lanczos_a);
            const float weight = w_x * w_y;

            const int s_x = (sample_x - (blockIdx.x * blockDim.x - padding)) + padding;
            const int s_y = (sample_y - (blockIdx.y * blockDim.y - padding)) + padding;

            if (s_x < 0 || s_x >= tile_w ||
                s_y < 0 || s_y >= tile_h ) continue;
            
            const float4 pix = shared_tile[s_y * tile_w + s_x];
            sum.x += pix.x * weight;
            sum.y += pix.y * weight;
            sum.z += pix.z * weight;
            total_weight += weight;
        }
    }

    uchar4 output;
    if (total_weight > 1e-6f) {
        output.x = saturate_uchar(sum.x / total_weight);
        output.y = saturate_uchar(sum.y / total_weight);
        output.z = saturate_uchar(sum.z / total_weight);
        output.w = 255;
    } else {
        const int nearest_x = __float2int_rd(src_x);
        const int nearest_y = __float2int_rd(src_y);
        float4 nearest = tex2D<float4>(tex_src, nearest_x + 0.5f, nearest_y + 0.5f);
        output.x = saturate_uchar(nearest.x);
        output.y = saturate_uchar(nearest.y);
        output.z = saturate_uchar(nearest.z);
        output.w = 255;
    }
    surf2Dwrite(output, surf_dst, dst_x * sizeof(uchar4), dst_y);
}

template <int scale_factor, int lanczos_a, int padding>
void cuda_lanczos_rs( const cv::cuda::GpuMat& src
                    , cv::cuda::GpuMat& dst
                    , cudaStream_t stream
                    ) {
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(scale_factor >= 1 && scale_factor <= 4);
    CV_Assert(lanczos_a >= 2 && lanczos_a <= 5);

    const cv::Size dst_size(src.cols * scale_factor, src.rows * scale_factor);
    dst.create(dst.size(), CV_8UC4);

    struct texture_wrapper {
        cudaArray* array = nullptr;
        cudaTextureObject_t tex = 0;

        void create(const cv::cuda::GpuMat& frame) {
            cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
            cudaMallocArray(&array, &channel_desc, frame.cols, frame.rows);
            cudaMemcpy2DToArray( array
                               , 0, 0
                               , frame.data, frame.step
                               , frame.cols * sizeof(uchar4), frame.rows
                               , cudaMemcpyDeviceToDevice
                               ) ;
            cudaResourceDesc res_desc{};
            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = array;

            cudaTextureDesc tex_desc{};
            tex_desc.addressMode[0] = cudaAddressModeClamp;
            tex_desc.addressMode[1] = cudaAddressModeClamp;
            tex_desc.filterMode = cudaFilterModeLinear;
            tex_desc.readMode = cudaReadModeNormalizedFloat;
            cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr);
        }

        __inline__
        ~texture_wrapper() {
            if (tex) cudaDestroyTextureObject(tex);
            if (array) cudaFreeArray(array);
        }
    } tex_wrapper;

    cv::cuda::GpuMat src_conv;

    cv::cuda::Stream cuda_stream = cv::cuda::StreamAccessor::wrapStream(stream);

    if (src.channels() == 3) cv::cuda::cvtColor(src, src_conv, cv::COLOR_BGR2BGRA, 4, cuda_stream);
    else if (src.channels() == 1) cv::cuda::cvtColor(src, src_conv, cv::COLOR_GRAY2BGRA, 4, cuda_stream);
    else src_conv.clone();

    tex_wrapper.create(src_conv);


    cudaSurfaceObject_t surf_dst = 0;
    {
        cudaResourceDesc res_desc{};
        res_desc.resType = cudaResourceTypePitch2D;
        res_desc.res.pitch2D.devPtr = dst.data;
        res_desc.res.pitch2D.desc = cudaCreateChannelDesc<uchar4>();
        res_desc.res.pitch2D.width = dst.cols;
        res_desc.res.pitch2D.height = dst.rows;
        res_desc.res.pitch2D.pitchInBytes = dst.step;
        cudaCreateSurfaceObject(&surf_dst, &res_desc);
    }

    dim3 block(16, 16);
    dim3 grid(
        (dst.cols + block.x - 1) / block.x,
        (dst.rows + block.y - 1) / block.y
    );

    const int tile_w = block.x + 2 * padding;
    const int tile_h = block.y + 2 * padding;
    const size_t shared_memory = tile_w * tile_h * sizeof(float4);

    lanczos_upscale_kernel<scale_factor, lanczos_a, padding><<<grid, block, shared_memory>>>( tex_wrapper.tex, surf_dst
                                                                                                                         , src.cols, src.rows
                                                                                                                         ) ;

    cudaDestroySurfaceObject(surf_dst);
}

template void cuda_lanczos_rs<2, 4, 3>(const cv::cuda::GpuMat&, cv::cuda::GpuMat&, cudaStream_t);
template void cuda_lanczos_rs<4, 4, 3>(const cv::cuda::GpuMat&, cv::cuda::GpuMat&, cudaStream_t);


#include <cuda_surface_types.h>

template <typename T>
__device__ __forceinline__
T clamp(T val, T min, T max) {
    if (val < min) return min;
    else if (val > max) return max;
    else return val;
}

__device__ __forceinline__
float lanczos_weight(float x, int a = 2) {
    if (x == 0.f) return 1.f;
    if (x < -a || x >= a) return 0.f;
    const float pi_x = M_PI * x;
    return (a * sinf(pi_x) * sinf(pi_x / a)) / (pi_x * pi_x);
}

template <int SCALE_FACTOR, int LANCZOS_RADIUS = 2>
__global__ void edge_aware_kenel( cudaSurfaceObject_t input_surf
                                , cudaSurfaceObject_t output_surf
                                , int in_width, int in_height
                                , float edge_thresh = 30.f
                                ) {
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= in_width * SCALE_FACTOR ||
        out_y >= in_height * SCALE_FACTOR ) return;
    
    const float x_in = (out_x + 0.5f) / SCALE_FACTOR - 0.5f;
    const float y_in = (out_y + 0.5f) / SCALE_FACTOR - 0.5f;

    __shared__ float4 shared_mem[32 +4][32 + 4];

    const int base_x = blockIdx.x * blockDim.x - 2;
    const int base_y = blockIdx.y * blockDim.y - 2;

    for (int dy = threadIdx.y; dy < blockDim.y + 4; dy += blockDim.y) {
        for (int dx = threadIdx.x; dx < blockDim.x + 4; dx += blockDim.x) {
            int x = base_x + dx;
            int y = base_y + dy;
            x = ::clamp(x, 0, in_width - 1);
            y = ::clamp(y, 0, in_height - 1);

            uchar4 pixel;
            surf2Dread(&pixel, input_surf, x * sizeof(uchar4), y);
            shared_mem[dy][dx] = make_float4(pixel.x, pixel.y, pixel.z, 0);
        }
    }
    __syncthreads();

    float edge_strength = 0.f;
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int sx = threadIdx.x + 2 + dx;
            int sy = threadIdx.y + 2 + dy;

            float gray = 0.299f * shared_mem[sy][sx].x
                       + 0.587f * shared_mem[sy][sx].y
                       + 0.114f * shared_mem[sy][sx].z;
            
            edge_strength += gray * expf(-(dx * dx + dy * dy) / 2.f);
        }
    }

    edge_strength = fabsf(edge_strength - 5.f) / 10.f;

    float adaptive_weight = 1.f + edge_strength * 2.f;
    
    float4 sum = make_float4(0, 0, 0, 0);
    float weight_sum = 0.f;

    const int x_start = floor(x_in - LANCZOS_RADIUS + 1);
    const int y_start = floor(y_in - LANCZOS_RADIUS + 1);

    for (int j = y_start; j <= y_start + LANCZOS_RADIUS * 2 - 1; ++j) {
        const float dy = y_in - j;
        const float wy = lanczos_weight(dy);
        
        for (int i = x_start; i <= x_start + LANCZOS_RADIUS * 2 - 1; ++i) {
            const float dx = x_in - i;
            const float wx = lanczos_weight(dx);

            const float w = wx * wy * adaptive_weight;

            uchar4 pixel;
            surf2Dread( &pixel, input_surf
                      , ::clamp(i, 0, in_width - 1) * sizeof(uchar4)
                      , ::clamp(j, 0, in_height - 1)
                      ) ;
        
            sum.x += pixel.x * w;
            sum.y += pixel.y * w;
            sum.z += pixel.z * w;
            weight_sum += w;
        }
    }

    const float4 result = {
        __saturatef(sum.x / weight_sum),
        __saturatef(sum.y / weight_sum),
        __saturatef(sum.z / weight_sum),
        0
    };

    uchar4 output = make_uchar4( result.x * 255
                               , result.y * 255
                               , result.z * 255
                               , 255
                               ) ;
    surf2Dwrite(output, output_surf, out_x * sizeof(uchar4), out_y);
}

void cuda_edge_aware_rs( const cv::cuda::GpuMat &src
                       , cv::cuda::GpuMat &dst
                       , int scale_factor
                       , cudaStream_t stream
                       ) {
    CV_Assert(!src.empty());

    cv::cuda::GpuMat src_converted, dst_converted;

    cv::cuda::Stream cuda_stream = cv::cuda::StreamAccessor::wrapStream(stream);
    if (src.type() != CV_8UC4) {
        switch (src.channels()) {
            case 1: cv::cuda::cvtColor(src, src_converted, cv::COLOR_GRAY2BGRA, 4, cuda_stream); break;
            case 3: cv::cuda::cvtColor(src, src_converted, cv::COLOR_BGR2BGRA, 4, cuda_stream); break;
            case 4: src.convertTo(src_converted, CV_8UC4, 1.f, 0, cuda_stream);
            default: CV_Error(cv::Error::StsBadArg, "Unsuppported input channels");
        }
    } else src_converted = src;

    const cv::Size dst_size( src_converted.cols * scale_factor
        , src_converted.rows * scale_factor
        ) ;
    dst_converted.create(dst_size, CV_8UC4);
    dst_converted.setTo(cv::Scalar::all(0), cuda_stream);

    cudaSurfaceObject_t src_surf = 0;
    cudaSurfaceObject_t dst_surf = 0;

    cudaResourceDesc src_desc = {};
    src_desc.resType = cudaResourceTypePitch2D;
    src_desc.res.pitch2D.devPtr = src_converted.data;
    src_desc.res.pitch2D.desc = cudaCreateChannelDesc<uchar4>();
    src_desc.res.pitch2D.width = src_converted.cols;
    src_desc.res.pitch2D.height = src_converted.rows;
    src_desc.res.pitch2D.pitchInBytes = src_converted.step;
    cudaCreateSurfaceObject(&src_surf, &src_desc);

    cudaResourceDesc dst_desc = {};
    dst_desc.resType = cudaResourceTypePitch2D;
    dst_desc.res.pitch2D.devPtr = dst_converted.data;
    dst_desc.res.pitch2D.desc = cudaCreateChannelDesc<uchar4>();
    dst_desc.res.pitch2D.width = dst_converted.cols;
    dst_desc.res.pitch2D.height = dst_converted.rows;
    dst_desc.res.pitch2D.pitchInBytes = dst_converted.step;
    // cudaError_t cuda_status;
    // cuda_status = cudaCreateSurfaceObject(&dst_surf, &dst_desc);
        // Error
    

    const dim3 block(32, 8);
    const dim3 grid(
        (src_converted.cols + block.x - 1) / block.x,
        (src_converted.rows + block.y - 1) / block.y
    );

    switch(scale_factor) {
        case 2: edge_aware_kenel<2><<<grid, block, 0, stream>>>(src_surf, dst_surf, src_converted.cols, src_converted.rows); break;
        case 4: edge_aware_kenel<4><<<grid, block, 0, stream>>>(src_surf, dst_surf, src_converted.cols, src_converted.rows); break;
        default: {
            cudaDestroySurfaceObject(src_surf);
            cudaDestroySurfaceObject(dst_surf);
            CV_Error(cv::Error::StsBadArg, "Unsupported scale factor!");
            break;
        }
    }

    if (dst.type() == src_converted.type()) {
        dst_converted.copyTo(dst, cuda_stream);
    } else {
        switch (src.channels()) {
            case 1: cv::cuda::cvtColor(dst_converted, dst, cv::COLOR_BGRA2GRAY, 0, cuda_stream); break;
            case 3: cv::cuda::cvtColor(dst_converted, dst, cv::COLOR_BGRA2BGR, 0, cuda_stream); break;
            default: dst_converted.copyTo(dst, cuda_stream);
        }
    }
}





// Accessible
constexpr int BLOCK_SIZE = 8;
constexpr int SEARCH_RADIUS = 15;
constexpr int GROUP_SIZE = 15;
constexpr float HRANGE = 30.f;
constexpr float SIGMA = 5.f;

struct WNNM_block_data {
    float pixels[BLOCK_SIZE][BLOCK_SIZE];
    int x, y;
};

template <typename T>
__device__
void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

__device__
float dot_product(const float* a, const float* b, int n) {
    float sum = 0.f;
    for (int i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

__device__
void scale_vector(float* vec, float scale, int n) {
    for (int i = 0; i < n; ++i) vec[i] *= scale;
}

#include <cuda_fp16.h>
__device__
float half2_dot(const __half2* a, const __half2* b, int len) {
    float sum = 0.f;
    for (int i = 0; i < len; ++i) {
        sum += __low2float(a[i]) * __low2float(b[i]);
        sum += __high2float(a[i]) * __high2float(b[i]);
    }
    return sum;
}

__global__ void wnnm_kernel( cudaSurfaceObject_t input
                           , cudaSurfaceObject_t output
                           , int width, int height
                           , int scale
                           , float sigma
                           , float hR
                           ) {
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int src_x = dst_x / scale;
    const int src_y = dst_y / scale;

    if (src_x >= width || src_y >= height) return;

    float center_block[BLOCK_SIZE][BLOCK_SIZE];
    for (int dy = 0; dy < BLOCK_SIZE; ++dy) {
        for (int dx = 0; dx < BLOCK_SIZE; ++dx) {
            uchar4 p;
            surf2Dread( &p
                      , input
                      , (src_x * scale + dx - BLOCK_SIZE / 2) * sizeof(uchar4)
                      , src_y * scale + dy - BLOCK_SIZE / 2
                      ) ;
            center_block[dy][dx] = 0.299f * p.x + 0.587 * p.y + 0.114f * p.z;
        }
    }

    WNNM_block_data similar_blocks[GROUP_SIZE];
    float distances[GROUP_SIZE];

    for (int i = 0; i < GROUP_SIZE; ++i) distances[i] = FLT_MAX;
    
    for (int dy = -SEARCH_RADIUS; dy <= SEARCH_RADIUS; ++dy) {
        for (int dx = -SEARCH_RADIUS; dx <= SEARCH_RADIUS; ++dx) {
            int x = src_x + dx;
            int y = src_y + dy;
            if (x < 0 || x >= width ||
                y < 0 || y >= width ) continue;
            
            float dist = 0.f;
            for (int by = 0; by <= BLOCK_SIZE; ++by) {
                for (int bx = 0; bx <= BLOCK_SIZE; ++bx) {
                    uchar4 p;
                    surf2Dread( &p, input
                              , (x * scale + bx - BLOCK_SIZE / 2) * sizeof(uchar4)
                              , y * scale + by - BLOCK_SIZE / 2
                              ) ;
                    float val = 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
                    dist += (val - center_block[by][bx]) * (val - center_block[by][bx]);
                }
            }

            if (dist < distances[GROUP_SIZE - 1]) {
                distances[GROUP_SIZE - 1] = dist;
                WNNM_block_data data;
                #pragma unroll
                for (int by = 0; by < BLOCK_SIZE; ++by) {
                    #pragma unroll 
                    for (int bx = 0; bx < BLOCK_SIZE; ++bx) {
                        data.pixels[by][bx] = center_block[by][bx];
                    }
                }
                data.x = x;
                data.y = y;
                similar_blocks[GROUP_SIZE - 1] = data;

                for (int i = GROUP_SIZE - 1; i > 0; --i) {
                    if (distances[i] < distances[i - 1]) {
                        ::swap(distances[i], distances[i - 1]);
                        ::swap(similar_blocks[i] ,similar_blocks[i - 1]);
                    }
                }
            }
        }
    }

    
    const int patch_dim = BLOCK_SIZE * BLOCK_SIZE;
    float matrix[GROUP_SIZE][patch_dim];

    #pragma unroll
    for (int i = 0; i < GROUP_SIZE; ++i) {
        #pragma unroll
        for (int y = 0; y < GROUP_SIZE; ++y) {
            #pragma unroll
            for (int x = 0; x < GROUP_SIZE; ++x) {
                matrix[i][y * BLOCK_SIZE + x] = similar_blocks[i].pixels[y][x];
            }
        }
    }

    const float tau = sigma * sqrtf(2 * logf(patch_dim));
    #pragma unroll
    for (int i = 0; i < GROUP_SIZE; ++i) {
        float sum = sqrtf(dot_product(matrix[i], matrix[i], patch_dim));
        sum = fmaxf(sum - tau, 0.f);
        scale_vector(matrix[i], sum / (sum + tau), patch_dim);
    }

    float weight_sum = 0.f;
    float final_pixle = 0.f;
    for (int i = 0; i < GROUP_SIZE; ++i) {
        float w = expf(-distances[i] / (hR * hR));
        final_pixle += w * matrix[i][threadIdx.y * BLOCK_SIZE + threadIdx.x];
        weight_sum += w;
    }
    final_pixle /= weight_sum;

    surf2Dwrite(make_uchar4(final_pixle, final_pixle, final_pixle, 255)
               , output
               , dst_x * sizeof(uchar4)
               , dst_y
               ) ;
}

void cuda_wnnm_rs( const cv::cuda::GpuMat &src
                 , cv::cuda::GpuMat &dst
                 , int scale_factor
                 , cudaStream_t stream
                 ) {
    CV_Assert(!src.empty());

    cv::cuda::GpuMat src_converted, dst_converted;

    cv::cuda::Stream cuda_stream = cv::cuda::StreamAccessor::wrapStream(stream);
    if (src.type() != CV_8UC4) {
        switch (src.channels()) {
            case 1: cv::cuda::cvtColor(src, src_converted, cv::COLOR_GRAY2BGRA, 4, cuda_stream); break;
            case 3: cv::cuda::cvtColor(src, src_converted, cv::COLOR_BGR2BGRA, 4, cuda_stream); break;
            case 4: src.convertTo(src_converted, CV_8UC4, 1.f, 0, cuda_stream);
            default: CV_Error(cv::Error::StsBadArg, "Unsuppported input channels");
        }
    } else src_converted = src;

    const cv::Size dst_size( src_converted.cols * scale_factor
                           , src_converted.rows * scale_factor
                           ) ;
    dst_converted.create(dst_size, CV_8UC4);
    dst_converted.setTo(cv::Scalar::all(0), cuda_stream);

    auto create_surfaces = [](const cv::cuda::GpuMat& mat) {
        if (!mat.isContinuous()) CV_Error(cv::Error::StsBadArg, "Require continuous memory for surface");
        
        cudaResourceDesc res_desc = {};
        
        res_desc.resType = cudaResourceTypePitch2D;
        res_desc.res.pitch2D.devPtr = mat.data;
        res_desc.res.pitch2D.desc = cudaCreateChannelDesc<uchar4>();
        res_desc.res.pitch2D.width = mat.cols;
        res_desc.res.pitch2D.height = mat.rows;
        res_desc.res.pitch2D.pitchInBytes = mat.step;

        cudaSurfaceObject_t surf = {};
        cudaCreateSurfaceObject(&surf, &res_desc);
        return surf;
    };

    cudaSurfaceObject_t input_surf = create_surfaces(src_converted);
    cudaSurfaceObject_t output_surf = create_surfaces(dst_converted);

    dim3 block(32, 8);
    dim3 grid(
        (dst_converted.cols + block.x - 1) / block.x,
        (dst_converted.rows + block.y - 1) / block.y
    );


    
    if (grid.x > 0 && grid.y > 0) {
        wnnm_kernel<<<grid, block, 0, stream>>>(input_surf, output_surf
                                                                        , src_converted.cols, src_converted.rows
                                                                        , scale_factor
                                                                        , SIGMA
                                                                        , HRANGE
                                                                        ) ;
    }
    cudaDestroySurfaceObject(input_surf);
    cudaDestroySurfaceObject(output_surf);

    if (dst.type() == src_converted.type()) {
        dst_converted.copyTo(dst, cuda_stream);
    } else {
        switch (src.channels()) {
            case 1: cv::cuda::cvtColor(dst_converted, dst, cv::COLOR_BGRA2GRAY, 0, cuda_stream); break;
            case 3: cv::cuda::cvtColor(dst_converted, dst, cv::COLOR_BGRA2BGR, 0, cuda_stream); break;
            default: dst_converted.copyTo(dst, cuda_stream);
        }
    }
}
