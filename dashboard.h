#ifndef DASHBOARD_H
#define DASHBOARD_H


#include "gpuinfo.h"
#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/photo.hpp"
#include "sr_cuda.h"
#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QTimer>
#include <QImage>
#include <QDebug>
#include <QPixmap>
#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <opencv2/opencv.hpp>
#include <QVector>
#include <QGridLayout>
#include <QScrollArea>
#include <QScrollBar>
#include <QWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QFormLayout>
#include <QCursor>
#include <QApplication>
#include <QDir>
#include <QMessageBox>
#include <QGroupBox>
#include <QComboBox>


#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include "iteminfo.h"

#include <QThread>
#include <QMutex>

#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <QThread>
#include <QMutex>
#include <QImage>
#include <qboxlayout.h>
#include <qcombobox.h>
#include <qframe.h>
#include <qgroupbox.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qlayoutitem.h>
#include <qmutex.h>

#include "denoise_cuda.h"
#include <unordered_map>

template <typename T>
struct processing_step;

template <>
struct processing_step<cv::Mat> {
    virtual void apply(cv::Mat& frame) = 0;
    virtual ~processing_step() = default;
};

template <>
struct processing_step<cv::cuda::GpuMat> {
    virtual void apply(cv::cuda::GpuMat& frame, cv::cuda::Stream& stream) = 0;
    virtual ~processing_step() = default;
};

template <typename T>
class color_convert_step;

template <>
class color_convert_step<cv::Mat>
    : public processing_step<cv::Mat> {
public:
    explicit color_convert_step(QMutex& mutex)
        : m_mutex(mutex) { }
    void apply(cv::Mat& frame) override {
        QMutexLocker locker(&m_mutex);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    }
private:
    QMutex& m_mutex;
};

template <>
class color_convert_step<cv::cuda::GpuMat>
    : public processing_step<cv::cuda::GpuMat> {
public:
    color_convert_step(QMutex& mutex)
        : m_mutex(mutex) { }
    void apply(cv::cuda::GpuMat& frame, cv::cuda::Stream& stream) override {
        QMutexLocker locker(&m_mutex);
        cv::cuda::cvtColor(frame, frame, cv::COLOR_BGR2RGB, 0, stream);
    }
private:
    QMutex& m_mutex;
};

template <typename T>
class contrast_adjust_step;

template <>
class contrast_adjust_step<cv::Mat>
    : public processing_step<cv::Mat> {
public:
    contrast_adjust_step(float& contrast, QMutex& mutex)
        : m_contrast(contrast)
        , m_mutex(mutex) { }
    void apply(cv::Mat& frame) override {
        QMutexLocker locker(&m_mutex);
        if (m_contrast != 1.f) 
            frame.convertTo(frame, -1, m_contrast);
    }
private:
    float& m_contrast;
    QMutex& m_mutex;
};

template <>
class contrast_adjust_step<cv::cuda::GpuMat>
    : public processing_step<cv::cuda::GpuMat> {
public:
    contrast_adjust_step(float& contrast, QMutex& mutex)
        : m_contrast(contrast)
        , m_mutex(mutex) { }
    void apply(cv::cuda::GpuMat& frame, cv::cuda::Stream& stream) override {
        QMutexLocker locker(&m_mutex);
        if (m_contrast != 1.0f) 
            cv::cuda::multiply(frame, cv::Scalar::all(m_contrast), frame, 1, -1, stream);
    }
private:
    float& m_contrast;
    QMutex& m_mutex;
};


enum class sharpen_type {
    LAPLACIAN,
    UNSHARP_MASK,
    EDGE_ENHANCE  
};

template <sharpen_type Type, typename T>
class sharpen_strength_adjust_step;

template <sharpen_type Type>
class sharpen_strength_adjust_step<Type, cv::Mat>
    : public processing_step<cv::Mat> { 
public:
    sharpen_strength_adjust_step(float& strength, sharpen_type& type, QMutex& mutex)
    : m_strength(strength)
    , m_type(type)
    , m_mutex(mutex) { }

    void apply(cv::Mat& frame) override {
        QMutexLocker locker(&m_mutex);
        if (m_strength > 0) {
            cv::Mat kernel = get_kernel();
            cv::filter2D(frame, frame, -1, kernel * m_strength);
        }
    }
private:
    cv::Mat get_kernel() const {
        switch (m_type) {
            case sharpen_type::LAPLACIAN:
                return (cv::Mat_<float>(3, 3) << 
                     0.f, -1.f,  0.f,
                    -1.f,  5.f, -1.f,
                     0.f, -1.f,  0.f);
            case sharpen_type::UNSHARP_MASK:
                return (cv::Mat_<float>(5, 5) << 
                       1.f,    4.f,    6.f,    4.f,    1.f,
                       4.f,   16.f,   24.f,   16.f,    4.f,
                       6.f,   24.f, -476.f,   24.f,    6.f,
                       4.f,   16.f,   24.f,   16.f,    4.f,
                       1.f,    4.f,    6.f,    4.f,    1.f) / -256.f;
            case sharpen_type::EDGE_ENHANCE:
                return (cv::Mat_<float>(3, 3) << 
                    -1.f, -1.f, -1.f,
                    -1.f,  9.f, -1.f,
                    -1.f, -1.f, -1.f);
            default:
                return cv::Mat::ones(1, 1, CV_32F);
        }
    }
    float& m_strength;
    sharpen_type& m_type;
    QMutex& m_mutex;
};

template <sharpen_type Type>
class sharpen_strength_adjust_step<Type, cv::cuda::GpuMat>
    : public processing_step<cv::cuda::GpuMat> { 
public:
    sharpen_strength_adjust_step(float& strength, sharpen_type& type, QMutex& mutex)
    : m_strength(strength)
    , m_type(type)
    , m_mutex(mutex) { 
        m_kernels.resize(3);
        
        cv::Mat base_kernel = (cv::Mat_<float>(3, 3) << 
                                0.f, -1.f,  0.f,
                               -1.f,  5.f, -1.f,
                                0.f, -1.f,  0.f);

        m_kernels[0] = cv::cuda::createLinearFilter(
            CV_8UC1, CV_8UC1,
            base_kernel
        );  // LAPLACIAN
        
        base_kernel = (cv::Mat_<float>(5, 5) << 
                        1.f,   4.f,    6.f,    4.f,   1.f,
                        4.f,  16.f,   24.f,   16.f,   4.f,
                        6.f,  24.f, -476.f,   24.f,   6.f,
                        4.f,  16.f,   24.f,   16.f,   4.f,
                        1.f,   4.f,    6.f,    4.f,   1.f) / -256.f;

        m_kernels[1] = cv::cuda::createLinearFilter(
            CV_8UC1, CV_8UC1,
            base_kernel
        );  // UNSHARP MASK
        
        base_kernel = (cv::Mat_<float>(3, 3) << 
                        -1.f, -1.f, -1.f,
                        -1.f,  9.f, -1.f,
                        -1.f, -1.f, -1.f);

        m_kernels[2] = cv::cuda::createLinearFilter(
            CV_8UC1, CV_8UC1,
            base_kernel
        );  // EDGE ENHANCE
    }

    void apply(cv::cuda::GpuMat& frame, cv::cuda::Stream& stream) override {
        QMutexLocker locker(&m_mutex);
        if (m_strength > 0.f) {
            std::vector<cv::cuda::GpuMat> channels;
            cv::cuda::split(frame, channels, stream);

            for (auto& ch: channels) {
                cv::cuda::GpuMat buffer;
                m_kernels[static_cast<int>(m_type)]->apply(ch, buffer, stream);
                cv::cuda::addWeighted(ch, 1.f + m_strength, buffer, -m_strength, 0.f, ch, -1.f, stream);
            }
            cv::cuda::merge(channels, frame, stream);
            stream.waitForCompletion();
        }
    }
private:    
    float& m_strength;
    sharpen_type& m_type;
    QMutex& m_mutex;
    std::vector<cv::Ptr<cv::cuda::Filter>> m_kernels;
};

template <typename T>
class gamma_adjust_step;

template <>
class gamma_adjust_step<cv::Mat>: public processing_step<cv::Mat> {
public:
    gamma_adjust_step(float& gamma, QMutex& mutex)
        : m_gamma(gamma)
        , m_mutex(mutex) { }

    void apply(cv::Mat& frame) override {
        QMutexLocker locker(&m_mutex);
        if (m_gamma != 1.f) {
            cv::Mat tmp;
            frame.convertTo(tmp, CV_32F, 1.f/255.f);
            cv::pow(tmp, m_gamma, tmp);
            tmp.convertTo(frame, CV_8U, 255.f);
        }
    }

private:
    float& m_gamma;
    QMutex& m_mutex;
};

template <>
class gamma_adjust_step<cv::cuda::GpuMat>
    : public processing_step<cv::cuda::GpuMat> {
public:
    gamma_adjust_step(float& gamma, QMutex& mutex)
        : m_gamma(gamma)
        , m_mutex(mutex) { }
    
    void apply(cv::cuda::GpuMat& frame, cv::cuda::Stream& stream) override {
        QMutexLocker locker(&m_mutex);
        
        cv::cuda::GpuMat tmp;
        frame.convertTo(tmp, CV_32F, 1.f/255.f, stream);
        cv::cuda::pow(tmp, m_gamma, tmp);
        tmp.convertTo(frame, CV_8U, 255.f, stream);
    }
private:
    float& m_gamma;
    QMutex& m_mutex;
};

template <typename T>
class brightness_adjust_step;

template <>
class brightness_adjust_step<cv::Mat>
    : public processing_step<cv::Mat> {
public:
    brightness_adjust_step(int& delta, QMutex& mutex)
        : m_delta(delta)
        , m_mutex(mutex) {     }

    void apply(cv::Mat& frame) override { 
        QMutexLocker locker(&m_mutex);
        if (m_delta != 0) 
            cv::add(frame, cv::Scalar(m_delta, m_delta, m_delta), frame);
    }

private:
    int& m_delta;
    QMutex& m_mutex;
};

#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
template <>
class brightness_adjust_step<cv::cuda::GpuMat>
    :public processing_step<cv::cuda::GpuMat> {
public:
    brightness_adjust_step(int& delta, QMutex& mutex) 
        : m_delta(delta)
        , m_mutex(mutex) { }

    void apply(cv::cuda::GpuMat& frame, cv::cuda::Stream& stream) override {
        QMutexLocker locker(&m_mutex);
        if (m_delta != 0) {
            cv::cuda::add (frame
                         , cv::Scalar(m_delta, m_delta, m_delta)
                         , frame
                         , cv::noArray()
                         , -1
                         , stream
                         ) ;
        }
    }
private:
    int& m_delta;
    QMutex& m_mutex;
};

template <typename T>
class saturation_ajust_step;

template <>
class saturation_ajust_step<cv::Mat>
    : public processing_step<cv::Mat> {
public:
    saturation_ajust_step(float& factor, QMutex& mutex)
        : m_factor(factor)
        , m_mutex(mutex) { }

    void apply(cv::Mat& frame) override {
        QMutexLocker locker(&m_mutex);
        if (m_factor != 1.f) {
            qDebug() << "saturation factor: " << m_factor;
            cv::Mat float_frame;
            frame.convertTo(float_frame, CV_32FC3, 1.f / 255.f);

            cv::parallel_for_(cv::Range(0, float_frame.rows), [&](const cv::Range& range) {
                for (int r = range.start; r < range.end; ++r) {
                    cv::Vec3f* ptr = float_frame.ptr<cv::Vec3f>(r);
                    for (int c = 0; c < float_frame.cols; ++c) {
                        cv::Vec3f& pixel = ptr[c];
                        float gray = 0.299f * pixel[2] + 0.587f * pixel[1] + 0.114f * pixel[0];
                        cv::Vec3f gray_vec(gray, gray, gray);
                        pixel = gray_vec + (pixel - gray_vec) * m_factor;
                    }
                }
            });
            float_frame.convertTo(frame, CV_8UC3, 255);
        }
    }
private:
    float& m_factor;
    QMutex& m_mutex;
};

template <>
class saturation_ajust_step<cv::cuda::GpuMat>
    : public processing_step<cv::cuda::GpuMat> {
public:
    saturation_ajust_step(float& factor, QMutex& mutex)
        : m_factor(factor)
        , m_mutex(mutex) { }

    void apply(cv::cuda::GpuMat& frame, cv::cuda::Stream& stream = cv::cuda::Stream::Null()) override {
        QMutexLocker locker(&m_mutex);
        if (m_factor != 1.f) {
            qDebug () << "Saturation:  " << m_factor;
            cv::cuda::GpuMat float_frame;
            frame.convertTo(float_frame, CV_32FC3, 1.f / 255.f, stream);

            cv::cuda::GpuMat gray_frame;
            cv::cuda::cvtColor(float_frame, gray_frame, cv::COLOR_BGR2GRAY, 0, stream);

            cv::cuda::GpuMat gray_frame_3c;
            cv::cuda::GpuMat channels[] = {gray_frame, gray_frame, gray_frame};
            cv::cuda::merge(channels, 3, gray_frame_3c, stream);

            cv::cuda::subtract(float_frame, gray_frame_3c, float_frame, cv::noArray(), -1, stream);
            cv::cuda::multiply(float_frame, m_factor, float_frame, 1.f, -1, stream);
            cv::cuda::add(float_frame, gray_frame_3c, float_frame, cv::noArray(), -1, stream);
            
            float_frame.convertTo(frame, CV_8UC3, 255, stream);
        }
    }
private:
    float& m_factor;
    QMutex& m_mutex;
};


template <typename T>
class exposure_adjust_step;

template <>
class exposure_adjust_step<cv::Mat>
    : public processing_step<cv::Mat> { 
public:
    exposure_adjust_step(float& gamma, float& gain, QMutex& mutex)
        : m_gamma(gamma)
        , m_gain(gain)
        , m_mutex(mutex) { }
    
    void apply(cv::Mat& frame) override {
        QMutexLocker locker(&m_mutex);
        if (m_gamma != 1.f && m_gain != 1.f) {
            cv::Mat float_frame;
            frame.convertTo(float_frame, CV_32FC3, 1.f / 255.f);
            cv::pow(float_frame, m_gamma, float_frame);

            float_frame *= m_gain;

            float_frame = cv::min(float_frame, 1.f);
            float_frame.convertTo(frame, CV_8UC3, 255);
        }
    }
private:
    float& m_gamma;
    float& m_gain;
    QMutex& m_mutex;
};

template <>
class exposure_adjust_step<cv::cuda::GpuMat>
    : public processing_step<cv::cuda::GpuMat> { 
public:
    exposure_adjust_step(float& gamma, float& gain, QMutex& mutex)
        : m_gamma(gamma)
        , m_gain(gain)
        , m_mutex(mutex) { }

    void apply(cv::cuda::GpuMat& frame, cv::cuda::Stream& stream = cv::cuda::Stream::Null()) override {
        QMutexLocker locker(&m_mutex);
        if (m_gamma != 1.f && m_gain != 1.f) {
            cv::cuda::GpuMat float_frame;
            frame.convertTo(float_frame, CV_32FC3, 1.f / 255.f, stream);
            cv::cuda::pow(float_frame, m_gamma, float_frame, stream);
            cv::cuda::multiply(float_frame, m_gain, float_frame, 1.f, -1, stream);
            cv::cuda::threshold(float_frame, float_frame, 1.f, 1.f, cv::THRESH_TRUNC, stream);
            float_frame.convertTo(frame, CV_8UC3, 255, stream);
        }
    }
private:
    float& m_gamma;
    float& m_gain;
    QMutex& m_mutex;
};

enum class denoise_type {
    NONE,
    GAUSSIAN,  
    BILATERAL, 
    NLMEANS,   
    MEDIAN,    
};

template <typename T>
class denoise_adjust_step;

template <>
class denoise_adjust_step<cv::Mat>
    : public processing_step<cv::Mat> {
public:
    denoise_adjust_step(denoise_type& type, float& strength, QMutex& mutex)
        : m_type(type)
        , m_strength(strength)
        , m_mutex(mutex) { }

    void apply(cv::Mat& frame) override {
        QMutexLocker locker(&m_mutex);
        if (m_type != denoise_type::NONE && m_strength > 0) {
            switch(m_type) {
                case denoise_type::GAUSSIAN:  
                    cv::GaussianBlur(frame, frame, cv::Size(5, 5), m_strength * 3);
                    break;
                case denoise_type::BILATERAL:   // fatal error
                    cv::bilateralFilter(frame, frame, 9, m_strength * 5, m_strength * 15);
                    break;
                case denoise_type::NLMEANS: 
                    cv::fastNlMeansDenoisingColored(frame, frame, m_strength * 10.f, m_strength * 3.f, 7 , 21);
                    break;
                case denoise_type::MEDIAN: 
                    cv::medianBlur(frame, frame, std::max(3, static_cast<int>(m_strength * 5) * 2 + 1));
                    break;
                default:
                    break;
            }
        }
    }
private:
    denoise_type& m_type;
    float& m_strength;
    QMutex& m_mutex;
};

template <>
class denoise_adjust_step<cv::cuda::GpuMat>
    : public processing_step<cv::cuda::GpuMat> {
public:
    denoise_adjust_step(denoise_type& type, float& strength, QMutex& mutex)
        : m_type(type)
        , m_strength(strength)
        , m_mutex(mutex) { }

    void apply(cv::cuda::GpuMat& frame, cv::cuda::Stream& stream = cv::cuda::Stream::Null()) override {
        QMutexLocker locker(&m_mutex);
        if (m_type != denoise_type::NONE && m_strength > 0) {
            switch(m_type) {
                case denoise_type::GAUSSIAN: {
                    cv::Ptr<cv::cuda::Filter> gaussian 
                        = cv::cuda::createGaussianFilter( frame.type()
                                                        , frame.type()
                                                        , cv::Size(5, 5)
                                                        , m_strength * 3.f
                                                        ) ;
                    gaussian->apply(frame, frame, stream);
                    break;
                }
                case denoise_type::MEDIAN: {
                    static const int max_supported_ksize = 7;
                    const int ksize = 3 + static_cast<int>(m_strength * (max_supported_ksize - 3));
                    cv::Ptr<cv::cuda::Filter> median = cv::cuda::createMedianFilter( frame.type()
                                                                                   , ksize | 1
                                                                                   ) ;
                    median->apply(frame, frame, stream);
                    break;
                }
                case denoise_type::BILATERAL: {
                    const float max_sigma_color = 75.f;
                    const float max_sigma_space = 25.f;
                    const bool use_lab = true;

                    const float sigma_color = m_strength * max_sigma_color;
                    const float sigma_space = m_strength * max_sigma_space;

                    cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

                    cv::cuda::GpuMat filtered(frame.size(), frame.type());

                    constexpr int RADIUS = 4;
                    cuda_bilateral_filter<RADIUS>( frame
                                                 , filtered
                                                 , sigma_color, sigma_space
                                                 , use_lab
                                                 , cuda_stream
                                                 ) ;
                    filtered.copyTo(frame, stream);
                    break;
                }
                case denoise_type::NLMEANS: {
                    constexpr int PATCH_SIZE = 5;
                    constexpr int SEARCH_WINDOW = 21;
                    const float max_h = 12.f;

                    const float h = (1.f - m_strength * 0.9f) * max_h;

                    cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
                    cv::cuda::GpuMat filtered;
                    cuda_nlmeans_filter<PATCH_SIZE, SEARCH_WINDOW>( frame
                                                                  , filtered
                                                                  , h
                                                                  , cuda_stream
                                                                  ) ;
                    filtered.copyTo(frame, stream);
                    break;
                }
                default: break;     // Not Supported Type
            }
        }
    }
private:
    denoise_type& m_type;
    float& m_strength;
    QMutex& m_mutex;
};

#include <opencv2/ximgproc.hpp>

template <typename T>
class deblur_step;

enum class sr_type {
    NONE,
    BILINEAR,
    BICUBIC,
    LANCZOS,
    EDGE_AWARE,
    WNNM
};

template <typename T>
class super_resolution_step;

template <>
class super_resolution_step<cv::Mat>
    : public processing_step<cv::Mat> {
public: 
    super_resolution_step(sr_type& type, int& scale, QMutex& mutex)
        : m_type(type)
        , m_scale(scale)
        , m_mutex(mutex) { }

    void apply(cv::Mat& frame) override {
        QMutexLocker locker(&m_mutex);
        if (m_type != sr_type::NONE) {
            cv::Mat processed;
            const cv::Size new_size(frame.cols * m_scale, frame.rows * m_scale);

            switch (m_type) {
                case sr_type::BILINEAR: cv::resize(frame, processed, new_size, 0, 0, cv::INTER_LINEAR); break;
                case sr_type::BICUBIC: cv::resize(frame, processed, new_size, 0, 0, cv::INTER_CUBIC); break;
                case sr_type::LANCZOS: cv::resize(frame, processed, new_size, 0, 0, cv::INTER_LANCZOS4); break;
                case sr_type::EDGE_AWARE: {
                    cv::Mat enhanced;
                    cv::edgePreservingFilter(frame, enhanced, cv::RECURS_FILTER);
                    cv::resize(enhanced, processed, new_size, cv::INTER_CUBIC);
                    break;
                }
                case sr_type::WNNM: break;
                default: processed = frame.clone();
            }
            frame = processed;
        }
    }
private:
    sr_type& m_type;
    int& m_scale;
    QMutex& m_mutex;
};

template <>
class super_resolution_step<cv::cuda::GpuMat>
    : public processing_step<cv::cuda::GpuMat> {
public:
    super_resolution_step(sr_type& type, int& scale, QMutex& mutex)
        : m_type(type)
        , m_scale(scale)
        , m_mutex(mutex) { }

    void apply(cv::cuda::GpuMat& frame, cv::cuda::Stream& stream = cv::cuda::Stream::Null()) override {
        QMutexLocker locker(&m_mutex);
        if (m_type != sr_type::NONE) {
            switch(m_type) {
                case sr_type::BILINEAR: {
                    break;
                }
                case sr_type::BICUBIC: break;
                case sr_type::EDGE_AWARE: {
                    cv::cuda::GpuMat upscaled;
                    cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
                    cuda_edge_aware_rs(frame, upscaled, 4, cuda_stream);
                    upscaled.copyTo(frame, stream);
                    break; 
                }
                case sr_type::WNNM: { 
                    cv::cuda::GpuMat upscaled;
                    cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
                    cuda_wnnm_rs(frame, upscaled, 4, cuda_stream);
                    upscaled.copyTo(frame, stream);
                    break;
                }
                case sr_type::LANCZOS: {                
                    cv::cuda::GpuMat src_4c;
                    cv::cuda::GpuMat dst_4c;
                    if (frame.channels() == 3) cv::cuda::cvtColor(frame, src_4c, cv::COLOR_BGR2BGRA, 4, stream);
                    else if (frame.channels() == 1) cv::cuda::cvtColor(frame, src_4c, cv::COLOR_GRAY2BGRA, 4, stream);
                    else src_4c = frame;
                    cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
                    cuda_lanczos_rs<2, 4, 3>(src_4c, dst_4c, cuda_stream);
                    if (frame.channels() == 3) {
                        // cv::cuda::cvtColor(dst_4c, frame, cv::COLOR_BGRA2BGR, 3, stream);
                    }
                    else if (frame.channels() == 1) {
                        cv::cuda::GpuMat gray_result;
                        cv::cuda::cvtColor(dst_4c, gray_result, cv::COLOR_BGRA2GRAY, 1, stream);
                        gray_result.copyTo(frame, stream);
                    } else dst_4c.copyTo(frame, stream);
                    break;
                }
                default: break;
            }
        }
    }
private:
    sr_type& m_type;
    int& m_scale;
    QMutex& m_mutex;
};

#ifdef ENABLE__SR_DL__
enum class sr_dl_type {
    NONE,
    EDSR,
    ESPCN,
    FSRCNN,
    LAPSRN,
};

template <typename T>
class super_resolution_dl_step;

template <>
class super_resolution_dl_step<cv::Mat>
    : public processing_step<cv::Mat> {
public:
    super_resolution_dl_step(sr_dl_type& type, int& scale, QMutex& mutex)
        : m_type(type)
        , m_scale(scale)
        , m_mutex(mutex) { }

    void apply(cv::Mat& frame) override {
        QMutexLocker locker(&m_mutex);
    }
private:
    sr_dl_type& m_type;
    int& m_scale;
    QMutex& m_mutex;
};

template <>
class super_resolution_dl_step<cv::cuda::GpuMat>
    : public processing_step<cv::cuda::GpuMat> {
public:
    super_resolution_dl_step(sr_dl_type& type, int& scale, QMutex& mutex)
        : m_type(type)
        , m_scale(scale)
        , m_mutex(mutex) { }

    void apply(cv::cuda::GpuMat& frame, cv::cuda::Stream& stream = cv::cuda::Stream::Null()) override {
        QMutexLocker locker(&m_mutex);
    }
private:
    sr_dl_type& m_type;
    int& m_scale;
    QMutex& m_mutex;
};

#endif

template <typename... Steps>
class processing_pipeline {
public:
    template <typename Step, typename... Args>
    void add_step(Args&&... args) {
        constexpr std::size_t index = get_step_index<Step>();
        std::get<index>(steps) = std::make_unique<Step>(std::forward<Args>(args)...);
    }
    void apply(cv::Mat& frame) {
        apply_impl(frame);
    }
    void apply(cv::cuda::GpuMat frame, cv::cuda::Stream& stream = cv::cuda::Stream::Null()) {
        apply_impl(frame, stream);
    }
private:
    template <typename Step>
    static constexpr std::size_t get_step_index() {
        return index_in_pack<Step, Steps...>();
    }

    template <typename T, typename U, typename... Us>
    static constexpr std::size_t index_in_pack() {
        if constexpr (std::is_same_v<T, U>) return 0;
        else return 1 + index_in_pack<T, Us...>();
    }
    
    template <typename T, typename... Args>
    void apply_impl(T& frame, Args&&... args) {
        (std::get<std::unique_ptr<Steps>>(steps)->apply(frame, std::forward<Args>(args)...), ...);
    }
    std::tuple<std::unique_ptr<Steps>...> steps;
};



class camera_capturer : public QThread {
    Q_OBJECT
public:
    explicit camera_capturer(int cam_idx = 0, bool use_cuda = false, QObject* parent = nullptr)
        : QThread(parent), m_cam_idx(cam_idx), m_use_cuda(use_cuda) {
        m_cpu_pipeline.add_step<color_convert_step<cv::Mat>>(m_mutex);
        m_cpu_pipeline.add_step<exposure_adjust_step<cv::Mat>>(m_exposure_gamma, m_exposure_gain, m_mutex);
        m_cpu_pipeline.add_step<contrast_adjust_step<cv::Mat>>(m_contrast, m_mutex);
        m_cpu_pipeline.add_step<brightness_adjust_step<cv::Mat>>(m_brightness, m_mutex);
        m_cpu_pipeline.add_step<saturation_ajust_step<cv::Mat>>(m_saturation, m_mutex);
        m_cpu_pipeline.add_step<sharpen_strength_adjust_step<sharpen_type::LAPLACIAN, cv::Mat>>(m_sharpen_strength, m_sharpen_type, m_mutex);
        m_cpu_pipeline.add_step<gamma_adjust_step<cv::Mat>>(m_gamma, m_mutex);
        m_cpu_pipeline.add_step<denoise_adjust_step<cv::Mat>>(m_denoise_type, m_denoise_strength, m_mutex);
        m_cpu_pipeline.add_step<super_resolution_step<cv::Mat>>(m_super_resolution_type, m_sr_scale, m_mutex);
        
        #ifdef ENABLE__SR_DL__
        // m_cpu_pipeline.add_step<super_resolution_dl_step<cv::Mat>>(m_super_resolution_dl_type, m_sr_dl_scale, m_mutex);
        #endif

        m_cuda_available = cv::cuda::getCudaEnabledDeviceCount() > 0;
        if (m_cuda_available) {
            m_cuda_pipeline.add_step<color_convert_step<cv::cuda::GpuMat>>(m_mutex);
            m_cuda_pipeline.add_step<exposure_adjust_step<cv::cuda::GpuMat>>(m_exposure_gamma, m_exposure_gain, m_mutex);
            m_cuda_pipeline.add_step<contrast_adjust_step<cv::cuda::GpuMat>>(m_contrast, m_mutex);
            m_cuda_pipeline.add_step<brightness_adjust_step<cv::cuda::GpuMat>>(m_brightness, m_mutex);
            m_cuda_pipeline.add_step<saturation_ajust_step<cv::cuda::GpuMat>>(m_saturation, m_mutex);
            m_cuda_pipeline.add_step<sharpen_strength_adjust_step<sharpen_type::LAPLACIAN, cv::cuda::GpuMat>>(m_sharpen_strength, m_sharpen_type, m_mutex);
            m_cuda_pipeline.add_step<gamma_adjust_step<cv::cuda::GpuMat>>(m_gamma, m_mutex);
            m_cuda_pipeline.add_step<denoise_adjust_step<cv::cuda::GpuMat>>(m_denoise_type, m_denoise_strength, m_mutex);
            m_cuda_pipeline.add_step<super_resolution_step<cv::cuda::GpuMat>>(m_super_resolution_type, m_sr_scale, m_mutex);
            #ifdef ENABLE__SR_DL__
            // m_cuda_pipeline.add_step<super_resolution_dl_step<cv::cuda::GpuMat>>(m_super_resolution_dl_type, m_sr_dl_scale, m_mutex);
            #endif
        }
    }

    void set_cuda_enabled(bool enable) {        // cuda
        QMutexLocker locker(&m_mutex);
        if (enable != m_use_cuda) {
            m_use_cuda = enable;
            if (enable && m_cuda_available) 
                cv::cuda::setDevice(0);
        }
    }

    void set_denoise_type(denoise_type type) {       // denoise
        QMutexLocker locker(&m_mutex);
        m_denoise_type = type;
    }
    
    void set_denoise_strength(float strength) {
        QMutexLocker locker(&m_mutex);
        m_denoise_strength = strength;
    }

    void set_contrast(float contrast) {         // contrast
        QMutexLocker locker(&m_mutex);
        m_contrast = contrast;
    }

    void set_gamma(float gamma) {               // gamma
        QMutexLocker locker(&m_mutex);
        m_gamma = gamma;
    }

    void set_brightness(int brightness) {       // brightness
        QMutexLocker locker(&m_mutex);
        m_brightness = brightness;
    }

    void set_saturation(float saturation) {     // saturation
        QMutexLocker locker(&m_mutex);
        m_saturation = saturation;
    }

    void set_sharpen_type(sharpen_type type) {  // sharpen_type
        QMutexLocker locker(&m_mutex);
        m_sharpen_type = type;
    }

    void set_sharpen_strength(float strength) { // sharpen_strength
        QMutexLocker locker(&m_mutex);
        m_sharpen_strength = strength;
    }
    
    void set_exp_gamma(float gamma) {
        QMutexLocker locker(&m_mutex);
        m_exposure_gamma = gamma;
    }

    void set_exp_gain(float gain) {
        QMutexLocker locker(&m_mutex);
        m_exposure_gain = gain;
    }

    void set_super_resolution_gr_type(sr_type type) {
        QMutexLocker  locker(&m_mutex);
        m_super_resolution_type = type;
    }
    
    void set_super_resolution_gr_scale(int scale) {
        QMutexLocker locker(&m_mutex);
        m_sr_scale = scale;
    }

    #ifdef ENABLE__SR_DL__
    void set_super_resolution_dl_type(sr_dl_type type) {
        QMutexLocker locker(&m_mutex);
        m_super_resolution_dl_type = type;
    }
    #endif

    void stop() {
        m_stop = true;
        wait();
    }

signals:
    void frame_captured(const QImage& img);
    void error_occurred(const QString& msg);

protected:
    void run() override {
        cv::VideoCapture cap{m_cam_idx};
        cv::cuda::GpuMat gpu_frame;

        if (!cap.isOpened()) {
            emit error_occurred(QString("无法打开摄像头: %1").arg(m_cam_idx));
            return;
        }

        while (!m_stop) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) break;

            bool use_cuda;
            {
                QMutexLocker locker(&m_mutex);
                use_cuda = m_use_cuda && m_cuda_available;
            }

            QImage output;
            try {
                if (use_cuda) {
                    gpu_frame.upload(frame, m_stream);
                    m_cuda_pipeline.apply(gpu_frame, m_stream);
                    cv::Mat result;
                    gpu_frame.download(result, m_stream);
                    m_stream.waitForCompletion();
                    output = QImage(result.data, result.cols, result.rows,
                                  result.step, QImage::Format_RGB888).copy();
                    qDebug() << "CUDA Enabled!";
                } else {
                    cv::Mat processed = frame.clone();
                    m_cpu_pipeline.apply(processed);
                    output = QImage(processed.data, processed.cols, processed.rows,
                                  processed.step, QImage::Format_RGB888).copy();
                }

                if (!output.isNull()) {
                    emit frame_captured(output);
                }
            } catch (const cv::Exception& e) {
                emit error_occurred(QString("处理错误: %1").arg(e.what()));
                if (m_use_cuda) {
                    QMutexLocker locker(&m_mutex);
                    m_use_cuda = false;
                    m_cuda_available = false;
                }
            }
        }
        cap.release();
    }

private:
    using CPU_pipeline = processing_pipeline<
        color_convert_step<cv::Mat>,
        exposure_adjust_step<cv::Mat>,
        contrast_adjust_step<cv::Mat>,
        brightness_adjust_step<cv::Mat>,
        saturation_ajust_step<cv::Mat>,
        sharpen_strength_adjust_step<sharpen_type::LAPLACIAN, cv::Mat>,
        gamma_adjust_step<cv::Mat>,
        denoise_adjust_step<cv::Mat>,
        super_resolution_step<cv::Mat>
        #ifdef ENABLE__SR_DL__
        // super_resolution_dl_step<cv::Mat>
        #endif
    >;

    using CUDA_pipeline = processing_pipeline<
        color_convert_step<cv::cuda::GpuMat>,
        exposure_adjust_step<cv::cuda::GpuMat>,
        contrast_adjust_step<cv::cuda::GpuMat>,
        brightness_adjust_step<cv::cuda::GpuMat>,
        saturation_ajust_step<cv::cuda::GpuMat>,
        sharpen_strength_adjust_step<sharpen_type::LAPLACIAN, cv::cuda::GpuMat>,
        gamma_adjust_step<cv::cuda::GpuMat>,
        denoise_adjust_step<cv::cuda::GpuMat>,
        super_resolution_step<cv::cuda::GpuMat>
        #ifdef ENABLE__SR_DL__
        // super_resolution_dl_step<cv::cuda::GpuMat>
        #endif
    >;

    CPU_pipeline m_cpu_pipeline;
    CUDA_pipeline m_cuda_pipeline;

    cv::cuda::Stream m_stream;
    int m_cam_idx;
    volatile bool m_stop = false;
    mutable QMutex m_mutex;
    
    bool m_use_cuda;    
    bool m_cuda_available = false;
    float m_contrast = 1.f;
    float m_gamma = 1.f;
    int m_brightness = 0;
    float m_saturation = 1.f;
    denoise_type m_denoise_type = denoise_type::BILATERAL;
    float m_denoise_strength = 0.f;
    
    sharpen_type m_sharpen_type = sharpen_type::LAPLACIAN;
    float m_sharpen_strength = 0.f; 
    float m_exposure_gamma = 1.f;
    float m_exposure_gain = 1.f;
    sr_type m_super_resolution_type = sr_type::NONE;
    int m_sr_scale = 2;
    #ifdef ENABLE__SR_DL__
    sr_dl_type m_super_resolution_dl_type = sr_dl_type::ESPCN;
    int m_sr_dl_scale = 4;
    #endif
};



#include <QSettings>


class Dashboard : public QWidget
{
    Q_OBJECT
public:
    explicit Dashboard(QWidget *parent = nullptr);
    ~Dashboard();
signals:


protected:
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private slots:
    bool saveItemInfo();
    void updateFrame(const QImage& img);
    
    // left side configuration
    void on_cuda_toggled(bool checked);                 // cuda
    void on_camera_change(int idx);                     // camera
    void on_camera_error_occur(const QString& msg);     // camera-error
    void on_contrast_changed(int index);                // contrast
    void on_gamma_changed(int index);                   // gamma
    void on_brightness_changed(int index);              // brightness
    void on_saturation_changed(int index);              // saturation
    void on_denoise_type_changed(int index);            // denoise-type
    void on_denoise_strength_changed(int index);        // denoise-strength
    
    // right side configuration
    void on_exp_gamma_changed(int index);               // exposure-gamma
    void on_exp_gain_changed(int index);                // exposure-gain
    void on_sharpen_type_changed(int index);            // sharpen-type
    void on_sharpen_strength_changed(int index);        // sharpen-strength
    #ifdef ENABLE__SR_DL__
    void on_sr_dl_changed(int index);                   // super resolution-deep learning
    #endif
    void on_sr_gr_changed(int index);                   // super resolution-general
    void on_sr_scale_changed(int index);                // super resolution scale

private:

    void init_cam_selector();
    void stop_current_cam();
    QVector<int> find_available_cams() const;

    std::unique_ptr<camera_capturer> m_cam;

    QElapsedTimer m_fps_timer;
    int m_frame_counter = 0;
    float m_current_fps = 0;

    void draw_fps_overlay(QPainter& painter);

    QSettings m_settings;
    bool m_cuda_enabled = false;

    float m_contrast = 1.f;
    
    QHBoxLayout* dashboard;
    // cameras and config
    QGroupBox* camera_and_config_panel;
    QVBoxLayout*  camera_and_config_panel_wrapper;
    // --camera
    QWidget* camera_panel;
    QVBoxLayout* camera_wrapper;
    QLabel* camera_label;
    // --config
    QGroupBox* camera_config_panel;
    QHBoxLayout* camera_config_layout_wrapper;
    // ----left
    QVBoxLayout* camera_left_config_layout;
    // ------enable cuda & camera select
    QHBoxLayout* enable_cuda_and_camera_select_wrapper;
    // ------enable cuda
    QHBoxLayout* check_cuda_enable_wrapper;
    QLabel* check_cuda_enable_label;
    QCheckBox* check_cuda_enable_checkbox;
    // ------camera select
    QHBoxLayout* select_camera_wrapper;
    QLabel* select_camera_label;
    QComboBox* select_camera_combobox;
    // ------contrast select
    QHBoxLayout* select_contrast_wrapper;
    QLabel* select_contrast_label;
    QComboBox* select_contrast_combobox;
    // ------gamma select
    QHBoxLayout* select_gamma_wrapper;
    QLabel* select_gamma_label;
    QComboBox* select_gamma_combobox;
    // ------brightness select
    QHBoxLayout* select_brightness_wrapper;
    QLabel* select_brightness_label;
    QComboBox* select_brightness_combobox;
    // ------saturation select
    QHBoxLayout* select_saturation_wrapper;
    QLabel* select_saturation_label;
    QComboBox* select_saturation_combobox;
    // ------denoise params 
    QHBoxLayout* select_denoise_params_wrapper;
    QGroupBox* select_denoise_params_box;
    QHBoxLayout* select_denoise_params_subwrapper;
    // ------denoise type select
    QHBoxLayout* select_denoise_type_wrapper;
    QLabel* select_denoise_type_label;
    QComboBox* select_denoise_type_combobox;
    // ------denoise strength select
    QHBoxLayout* select_denoise_strength_wrapper;
    QLabel* select_denoise_strength_label;
    QComboBox* select_denoise_strength_combobox;

    // ----right
    QVBoxLayout* camera_right_config_layout;
    // ------exposure select
    QHBoxLayout* select_exposure_params_wrapper;
    QGroupBox* select_exposure_params_box;
    QHBoxLayout* select_exposure_params_subwrapper;
    // ------exposure gamma select
    QHBoxLayout* select_exposure_gamma_wrapper;
    QLabel* select_exposure_gamma_label;
    QComboBox* select_exposure_gamma_combobox;
    // ------exposure gain select
    QHBoxLayout* select_exposure_gain_wrapper;
    QLabel* select_exposure_gain_label;
    QComboBox* select_exposure_gain_combobox;
    // ------sharpen params
    QHBoxLayout* select_sharpen_params_wrapper;
    QGroupBox* select_sharpen_params_box;
    QHBoxLayout* select_sharpen_params_subwrapper;
    // ------sharpen type select
    QHBoxLayout* select_sharpen_type_wrapper;
    QLabel* select_sharpen_type_label;
    QComboBox* select_sharpen_type_combobox;
    // ------sharpen strength select
    QHBoxLayout* select_sharpen_strength_wrapper;
    QLabel* select_sharpen_strength_label;
    QComboBox* select_sharpen_strength_combobox;
    // ------super resolution params
    QHBoxLayout* select_super_resolution_params_wrapper;
    QGroupBox* select_super_resolution_params_box;
    QHBoxLayout* select_super_resolution_params_subwrapper;
    #ifdef ENABLE__SR_DL__
    // ------super resolution by dl params
    QHBoxLayout* select_super_resolution_dl_params_wrapper;
    QLabel* select_super_resolution_dl_label;
    QComboBox* select_super_resolution_dl_combobox;
    #endif
    // ------super resolution by gr params
    QHBoxLayout* select_super_resolution_gr_params_wrapper;
    QLabel* select_super_resolution_gr_label;
    QComboBox* select_super_resolution_gr_combobox;
    // ------super resolution scale
    QHBoxLayout* select_super_resolution_scale_wrapper;
    QLabel* select_super_resolution_scale_label;
    QComboBox* select_super_resolution_scale_combobox;


    // item editor and gallery
    QGroupBox* editor_and_gallery_panel;



    QLabel* ItemImage;
    QLabel* classIdLabel;
    QLineEdit* ItemName;
    QLineEdit* ItemLength;
    QLineEdit* ItemWidth;
    QLineEdit* ItemHeight;

    QPushButton* save;


    QScrollArea* scrollArea;
    QWidget* scrollWidget;
    QGridLayout* galleryLayout;

    QVector<ItemInfo*> Items;

    QImage cvMatToImage(const cv::Mat& mat);

    QPoint selectionStart;
    QPoint selectionEnd;
    bool isSelecting = false;
    bool isDragging = false;
    QPoint dragOffset;

    QPoint mapToFrame(const QPoint& widgetPoint);
    QRect mapRectToFrame(const QRect& widgetRect);

    void
    addRegionToGallery( const QPixmap& pixmap
                      , const QString& id = ""
                      , const QString& name = ""
                      , const QString& length = ""
                      , const QString& width = ""
                      , const QString& height = ""
                      );

    // void updateUserArea(ItemInfo* item);

    void updateItemImage();

    QVector<QRect> selectedRects;
    QRect currentSelection;
    QImage current_cap_frame;

    bool isHoveringOverSelection = false;
    void onItemNameChanged(const QString& text);



};


#endif // DASHBOARD_H
