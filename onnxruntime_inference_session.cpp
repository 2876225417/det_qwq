#include "onnxruntime_inference_session.h"
#include <QDebug>

onnxruntime_inference_session::onnxruntime_inference_session():
      _onnx_model_file("yolov8m.onnx")
    , _class_file("class.txt"){
}

onnxruntime_inference_session::~onnxruntime_inference_session(){

}



cv::Mat onnxruntime_inference_session::_infer(const cv::Mat& in_mat){

    OrtSessionOptionsAppendExecutionProvider_CUDA(_session_options, 0);
    _session_options.DisableCpuMemArena();
    _session_options.SetIntraOpNumThreads(1);
    _session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    Ort::Session* _session = new Ort::Session(
        *_env
        //, (std::wstring(_onnx_model_file.begin(), _onnx_model_file.end())).c_str()
        , _onnx_model_file.c_str()
        , _session_options);

    Ort::AllocatorWithDefaultOptions        _allocator;



    _num_input_nodes    = _session->GetInputCount();
    _num_output_nodes   = _session->GetOutputCount();
    _labels = _read_class_names("class.txt");

    std::cout << "onnxruntime inference try to use GPU Device\n";
    _input_node_names.reserve(_num_input_nodes);

    // 获取输入信息
    int input_w_ = 0;
    int input_h_ = 0;
    for(int i = 0; i < _num_input_nodes; i++){
        auto input_name_                = _session->GetInputNameAllocated(i, _allocator);
        Ort::TypeInfo input_type_info_  = _session->GetInputTypeInfo(i);
        auto input_tensor_info_         = input_type_info_.GetTensorTypeAndShapeInfo();
        auto input_dims_                = input_tensor_info_.GetShape();
        input_w_                        = input_dims_[3];
        input_h_                        = input_dims_[2];
        _input_node_names.push_back(input_name_.get());
        std::cout << "Inout format: NxCxHxW = "
                  << input_dims_[0] << "x"
                  << input_dims_[1] << "x"
                  << input_dims_[2] << "x"
                  << input_dims_[3] << '\n';
    }

    Ort::TypeInfo   output_type_info    = _session->GetOutputTypeInfo(0);
    auto            output_tensor_info_ = output_type_info.GetTensorTypeAndShapeInfo();
    auto            output_dims_        = output_tensor_info_.GetShape();
    int             output_h_           = output_dims_[1];
    int             output_w_           = output_dims_[2];
    std::cout << "output format: HxW = "
              << output_dims_[1] << "x"
              << output_dims_[2] << '\n';
    for(int i = 0; i < _num_output_nodes; i++){
        auto out_name_ = _session->GetOutputNameAllocated(i, _allocator);
        _output_node_names.push_back(out_name_.get());
    }
    std::cout << "input: " << _input_node_names[0]
              << " output: " << _output_node_names[0] << '\n';

    cv::Mat in_frame = in_mat;
    if(in_frame.empty()){
        std::cerr << "Error: can not resolve image mat";
        return cv::Mat();
    }

    int64 start_ = cv::getTickCount();

    int                                 w_              = in_frame.cols;
    int                                 h_              = in_frame.rows;
    int                                 max_            = std::max(h_, w_);
    cv::Mat                             image_          = cv::Mat::zeros(cv::Size(max_, max_), CV_8UC3);
    float                               x_factor_       = image_.cols / static_cast<float>(input_w_);
    float                               y_factor_       = image_.rows / static_cast<float>(input_h_);
    cv::Rect                            roi_(0, 0, w_, h_);
    in_frame.copyTo(image_(roi_));
    cv::Mat                             blob_           = cv::dnn::blobFromImage(image_, 1 / 255.0, cv::Size(input_w_, input_h_), cv::Scalar(0, 0, 0), true, false);
    size_t                              tpixels_        = input_h_ * input_w_ * 3;
    std::array<int64_t, 4>              input_shape_info_{ 1, 3, input_h_, input_w_ };
    auto                                allocator_info_ = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value                          input_tenosr_   = Ort::Value::CreateTensor<float>(allocator_info_, blob_.ptr<float>(), tpixels_, input_shape_info_.data(), input_shape_info_.size());
    const std::array<const char*, 1>    input_names_    = { _input_node_names[0].c_str() };
    const std::array<const char*, 1>    output_names_   = { _output_node_names[0].c_str() };
    std::vector<Ort::Value> ort_outputs_;
    try{
        ort_outputs_ = _session->Run(
            Ort::RunOptions{ nullptr }
            , input_names_.data()
            , &input_tenosr_
            , 1
            , output_names_.data()
            , output_names_.size()
            );
    } catch(std::exception& e){
        std::endl(std::cout << e.what());
    }

    const float*            pdata_      = ort_outputs_[0].GetTensorMutableData<float>();
    cv::Mat                 dout_(output_h_, output_w_, CV_32F, (float*)pdata_);
    cv::Mat                 det_output_ = dout_.t();
    std::vector<cv::Rect>   boxes_;
    std::vector<int>        class_ids_;
    std::vector<float>      confidences_;

    for(int i = 0; i < det_output_.rows; i++){
        cv::Mat     classes_scores_ = det_output_.row(i).colRange(4, 84);
        cv::Point   class_id_point_;
        double      score_;
        cv::minMaxLoc(classes_scores_, 0, &score_, 0, &class_id_point_);

        if(score_ > 0.25){
            float       cx_         = det_output_.at<float>(i, 0);
            float       cy_         = det_output_.at<float>(i, 1);
            float       ow_         = det_output_.at<float>(i, 2);
            float       oh_         = det_output_.at<float>(i, 3);
            int         x           = static_cast<int>((cx_ - 0.5 * ow_) * x_factor_);
            int         y           = static_cast<int>((cy_ - 0.5 * oh_) * y_factor_);
            int         width_      = static_cast<int>(ow_ * x_factor_);
            int         height_     = static_cast<int>(oh_ * y_factor_);
            cv::Rect    box_;
            box_.x      = x;
            box_.y      = y;
            box_.width  = width_;
            box_.height = height_;
            boxes_.push_back(box_);
            class_ids_.push_back(class_id_point_.x);
            confidences_.push_back(score_);
        }
    }

    std::vector<int> indexes_;
    cv::dnn::NMSBoxes(boxes_, confidences_, 0.25, 0.45, indexes_);
    for(size_t i = 0; i < indexes_.size(); i++){
        int index   = indexes_[i];
        int idx     = class_ids_[index];
        cv::rectangle(in_frame, boxes_[index], cv::Scalar(0, 0, 255), 2, 8);
        cv::rectangle( in_frame
                     , cv::Point(boxes_[index].tl().x, boxes_[index].tl().y - 20)
                     , cv::Point(boxes_[index].br().x, boxes_[index].tl().y)
                     , cv::Scalar(0, 255, 255), -1
                     ) ;
        putText( in_frame
               , (*_labels)[idx]
               , cv::Point(boxes_[index].tl().x, boxes_[index].tl().y)
               , cv::FONT_HERSHEY_PLAIN
               , 2.0
               , cv::Scalar(255, 0, 0)
               , 2
               , 8
               ) ;
    }

    float t = (cv::getTickCount() - start_) / static_cast<float>(cv::getTickFrequency());
    cv::putText(
         in_frame
        , cv::format("FPS: %.2f", 1.0 / t)
        , cv::Point(20, 48)
        , cv::FONT_HERSHEY_PLAIN
        , 2.0
        , cv::Scalar(255, 0, 0)
        , 2
        , 8
        ) ;

    delete _session;

    qDebug() << "All released";
    return in_frame;
}




std::optional<std::vector<std::string>> onnxruntime_inference_session::_read_class_names(const std::string& in_class_file){
    std::ifstream file_(in_class_file);
    if(!file_){
        std::cerr << "Error opening class names file: " << in_class_file << '\n';
        return {};
    }

    std::vector<std::string> class_names;
    std::string line;
    while(std::getline(file_, line))
        class_names.emplace_back(std::move(line));

    return class_names;
}







