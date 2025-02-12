#ifndef ONNXRUNTIME_INFERENCE_SESSION_H
#define ONNXRUNTIME_INFERENCE_SESSION_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <filesystem>

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



private:
    Ort::Env*                                _env = new Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov8-onnx");
    Ort::SessionOptions                     _session_options;




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
