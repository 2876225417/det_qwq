#ifndef ONNXRUNTIME_INFERENCE_SESSION_H
#define ONNXRUNTIME_INFERENCE_SESSION_H

#include "opencv2/core/types.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <qbytearrayview.h>
#include <qvariant.h>
#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <filesystem>
#include <QRect>


struct detection_result {
    QVector<QRect> boxes;
    QVector<int> class_ids;
    QVector<float> scores;
};

enum class border_color {
    red = 0,
    green,
    blue,
    white,
    black,
    cyan
};

enum class font_type {
    simplex = 0,
    plain,
    duplex,
    complex,
    triplex,
    complex_small
 };

enum class font_color { 
    high_contrast = 0,
    pure_white,
    neon_green,
    warning_yellow,
    signal_red
};

enum class filling_color {
    None = 0,
    dark_overlay,
    light_overlay,
    danger_highlight,
    info_highlight
};

enum class label_position {
    top_left = 0,
    top_right = 1,
    bottom_left = 2,
    bottom_right = 3,
    center = 4
};



class onnxruntime_inference_session
{
public:
    onnxruntime_inference_session();
    ~onnxruntime_inference_session();

public:
    void set_class_path(const std::string in_class_path);
    void set_onnx_model_path(const std::string in_onnx_model_path);

    cv::Mat _infer(const cv::Mat& in_mat);
    cv::Mat _infer_frame(
          cv::Mat&  in_frame
        , int       in_input_w
        , int       in_input_h
        , int       in_output_w
        , int       in_output_h
        , float     in_start
        );

    std::vector<cv::Rect> m_boxes;
    std::vector<int> m_class_ids;
    std::vector<float> m_scores;

    const std::vector<cv::Rect>& get_boxes() const { return m_boxes; }
    const std::vector<int>& get_class_ids() const { return m_class_ids; }
    const std::vector<float>& get_scores() const { return m_scores; }
    

    QStringList get_labels() const {
        QStringList labels;
        if (_labels.has_value()) {
            for (const auto& s: *_labels)
                labels << QString::fromStdString(s);
        }
        return labels;
    }
    void set_border_color(border_color bc) { m_border_color = bc; }
    void set_font_type(font_type ft) { m_font_type = ft; }
    void set_font_color(font_color fc) { m_font_color = fc; }
    void set_filling_color(filling_color fic) { m_filling_color = fic; }



private:
    Ort::Env*                                _env = new Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov8-onnx");
    Ort::SessionOptions                     _session_options;

    // draw config
    border_color m_border_color = border_color::black;
    font_type m_font_type = font_type::complex;
    font_color m_font_color = font_color::high_contrast;
    filling_color m_filling_color = filling_color::danger_highlight;


    cv::Point calc_label_position() {

    }

    cv::Scalar get_border_color() {
        static std::map<border_color, cv::Scalar> mapper {
            {border_color::red, cv::Scalar(0, 0, 255)},
            {border_color::green, cv::Scalar(0, 255, 0)},
            {border_color::blue, cv::Scalar(255, 0, 0)},
            {border_color::white, cv::Scalar(255, 255, 255)},
            {border_color::black, cv::Scalar(0, 0, 0)},
            {border_color::cyan, cv::Scalar(255, 255, 0)}
        };
        return mapper[m_border_color];
    }

    int get_font_type() {
        static std::map<font_type, cv::HersheyFonts> mapper {
            {font_type::simplex, cv::FONT_HERSHEY_SIMPLEX},
            {font_type::plain, cv::FONT_HERSHEY_PLAIN},
            {font_type::duplex, cv::FONT_HERSHEY_DUPLEX},
            {font_type::complex, cv::FONT_HERSHEY_COMPLEX},
            {font_type::triplex, cv::FONT_HERSHEY_TRIPLEX},
            {font_type::complex_small, cv::FONT_HERSHEY_COMPLEX_SMALL}
        };
        return mapper[m_font_type];
    }

    cv::Scalar get_font_color() {
        static std::map<font_color, cv::Scalar> mapper {
            {font_color::high_contrast, cv::Scalar(255, 255 ,255)},
            {font_color::pure_white, cv::Scalar(25, 255, 255)},
            {font_color::neon_green, cv::Scalar(0, 255, 0)},
            {font_color::warning_yellow, cv::Scalar(0, 255, 255)},
            {font_color::signal_red, cv::Scalar(0, 0, 255)}
        };
        return mapper[m_font_color];
    }

    cv::Scalar get_filling_color() {
        static std::map<filling_color, cv::Scalar> mapper {
            {filling_color::None, cv::Scalar(0, 0, 0, 0)},
            {filling_color::dark_overlay, cv::Scalar(0, 0, 0, 128)},
            {filling_color::light_overlay, cv::Scalar(255, 255, 255, 128)},
            {filling_color::danger_highlight, cv::Scalar(0, 0, 255, 64)},
            {filling_color::info_highlight, cv::Scalar(255, 0, 0, 64)}
        };
        return mapper[m_filling_color];
    }

    void draw_border( cv::Mat& frame, std::string label, cv::Rect point) { 
        cv::rectangle(frame, point, get_border_color(), 2, 8);
        cv::rectangle( frame
                     , cv::Point{point.tl().x, point.tl().y}
                     , cv::Point{point.br().x, point.tl().y}
                     , get_border_color()
                     , -1
                     ) ;
        
        cv::putText( frame, label
                   , cv::Point{point.tl().x, point.tl().y}
                   , get_font_type(), 2.0
                   , get_border_color(), 2, 8
                   ) ;
    }

    

    // onnxruntime_inference_session(const onnxruntime_inference_session&) = delete;
    // onnxruntime_inference_session& operator=(const onnxruntime_inference_session&) = delete;

    std::optional<std::vector<std::string>> _read_class_names(const std::string& in_class_file);



    size_t                                  _num_input_nodes;
    size_t                                  _num_output_nodes;
    std::optional<std::vector<std::string>> _labels;


    std::string                             _onnx_model_file;
    std::string                             _class_file;
    std::vector<std::string>                _input_node_names;
    std::vector<std::string>                _output_node_names;
};

#endif // ONNXRUNTIME_INFERENCE_SESSION_H
