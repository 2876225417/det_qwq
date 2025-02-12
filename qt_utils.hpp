#include <QDebug>  // 用于打印调试信息
#include <opencv2/opencv.hpp>
#include <optional>
#include <QImage>
#include <opencv2/cudaimgproc.hpp>

namespace qt_utils {
    cv::Mat get_continuous_mat(const cv::Mat& mat){
        if(mat.isContinuous())
            return mat;
        cv::Mat tmp;
        mat.copyTo(tmp);
        return tmp;
    }

    QImage create_QImage_from_BGR(const cv::Mat& mat){
        cv::Mat tmp = get_continuous_mat(mat);
        cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);
        return QImage( tmp.data
                     , tmp.cols
                     , tmp.rows
                     , tmp.step
                     , QImage::Format_RGB888
                     );
    }

    QImage create_QImage_from_gray(const cv::Mat& mat){
        return QImage( mat.data
                     , mat.cols
                     , mat.rows
                     , mat.step
                     , QImage::Format_Grayscale8
                     );
    }

    QImage create_QImage_from_RGBA(const cv::Mat& mat){
        cv::Mat tmp = get_continuous_mat(mat);
        if(!mat.isContinuous())
            cv::cvtColor(tmp, tmp, cv::COLOR_BGRA2RGBA);
        return QImage( tmp.data
                     , tmp.cols
                     , tmp.rows
                     , tmp.step
                     , QImage::Format_RGBA8888
                     );
    }

    cv::Mat convert_16bit_to_8bit(const cv::Mat& mat){
        cv::Mat normalized;
        cv::normalize( mat
                     , normalized
                     , 0
                     , 255
                     , cv::NORM_MINMAX
                     );
        normalized.convertTo(normalized, CV_8UC3);
        return normalized;
    }

    template<typename T>
    std::optional<QImage> mat_to_image(const cv::Mat& mat);

    template<>
    std::optional<QImage> mat_to_image<cv::Vec3b>(const cv::Mat& mat){
        if(mat.type() == CV_8UC3)
            return create_QImage_from_BGR(mat);
        else if(mat.type() == CV_16UC3){
            cv::Mat img_8bit = convert_16bit_to_8bit(mat);
            return create_QImage_from_BGR(img_8bit);
        }
        return std::nullopt;
    }

    template<>
    std::optional<QImage> mat_to_image<uchar>(const cv::Mat& mat){
        if(mat.type() == CV_8UC1)
            return create_QImage_from_gray(mat);
        else if(mat.type() == CV_16UC1){
            cv::Mat img_8bit = convert_16bit_to_8bit(mat);
            return create_QImage_from_gray(img_8bit);
        }
        return std::nullopt;
    }

    template<>
    std::optional<QImage> mat_to_image<cv::Vec4b>(const cv::Mat& mat){
        if(mat.type() == CV_8UC4)
            return create_QImage_from_RGBA(mat);
        return std::nullopt;
    }

    std::optional<QImage> mat_to_image(const cv::Mat& mat){
        if(mat.empty()){
            qDebug() << "Empty Mat!";
            return std::nullopt;
        }

        switch(mat.type()){
            case CV_8UC3: return mat_to_image<cv::Vec3b>(mat);
            case CV_8UC1: return mat_to_image<uchar>(mat);
            case CV_8UC4: return mat_to_image<cv::Vec4b>(mat);
            case CV_16UC3: return mat_to_image<cv::Vec3b>(mat);
            case CV_16UC1: return mat_to_image<uchar>(mat);
            default:
                qDebug() << "Unsupported Mat Type!";
                return std::nullopt;
        }
    }
}
