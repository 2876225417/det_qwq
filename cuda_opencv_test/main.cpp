#include <QApplication>
#include <QMainWindow>
#include <iostream>

#include "mainwindow.h"

// #include "onnxruntime_inference_session.h"

#include <QDebug>

#include <QProcess>


#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/dnn.hpp>

// CPU版本实现
void dl_sr_cpu(cv::Mat& frame, const std::string& model_path) {
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow(model_path);
    // BGR -> YCrCb 颜色空间转换
    cv::Mat ycc;
    cv::cvtColor(frame, ycc, cv::COLOR_BGR2YCrCb);
    
    // 分离Y通道（单通道）
    std::vector<cv::Mat> channels(3);
    cv::split(ycc, channels);
    cv::Mat y_channel = channels[0];
    
    // 预处理（单通道输入）
    cv::Mat blob;
    cv::dnn::blobFromImage(y_channel, blob, 1.0/255.0,
                          cv::Size(), 
                          cv::Scalar(0), 
                          false);  // 注意第三个参数设为false保持单通道
    
    net.setInput(blob);
    cv::Mat output = net.forward();

    // 后处理（保持YCrCb格式）
    output = 255.0f * output.reshape(1, output.size[2]);
    output.convertTo(output, CV_8U);
    
    // 合并通道
    std::vector<cv::Mat> sr_channels{output, channels[1], channels[2]};
    cv::merge(sr_channels, frame);
    cv::cvtColor(frame, frame, cv::COLOR_YCrCb2BGR);
}


// CUDA加速版本实现
// void dl_sr_cuda(cv::Mat& frame, const std::string& model_path) {
//     cv::dnn::Net net = cv::dnn::readNetFromTensorflow(model_path);
//     net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
//     net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

//     cv::cuda::GpuMat gpu_frame, gpu_blob, gpu_output;
//     cv::cuda::Stream stream;
    
//     // 转换为GPU内存对象
//     gpu_frame.upload(frame, stream);
    
//     // 在GPU上执行blob转换
//     cv::cuda::cvtColor(gpu_frame, gpu_frame, cv::COLOR_BGR2RGB, 0, stream);
//     cv::cuda::resize(gpu_frame, gpu_frame, cv::Size(), 1.0/4, 1.0/4, cv::INTER_CUBIC, stream);
//     gpu_frame.convertTo(gpu_blob, CV_32FC3, 1.0/255.0, stream);
    
//     // 构建4D blob [NCHW]
//     cv::cuda::GpuMat blob_4d(gpu_blob.rows * gpu_blob.cols * 3, 1, CV_32F, 
//                            gpu_blob.ptr(), gpu_blob.step);
//     blob_4d = blob_4d.reshape(1, {1, 3, gpu_blob.rows, gpu_blob.cols});

//     // 推理
//     net.setInput(blob_4d);
//     gpu_output = net.forward("", stream); // 指定输出层名称（模型相关）

//     // 后处理
//     cv::cuda::resize(gpu_output.reshape(3, gpu_output.size[2]), 
//                     gpu_frame, 
//                     cv::Size(frame.cols*4, frame.rows*4), 
//                     0, 0, cv::INTER_CUBIC, stream);
    
//     // 下载结果
//     gpu_frame.download(frame, stream);
//     stream.waitForCompletion();
// }

int main() {
    cv::Mat image = cv::imread("star.jpg");
    if(image.empty()) return -1;

    // CPU处理
    cv::Mat cpu_result = image.clone();
    dl_sr_cpu(cpu_result, "models/ESPCN_x4.pb");
    cv::imwrite("result_cpu.jpg", cpu_result);

    // CUDA处理
    // if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
    //     cv::Mat cuda_result;
    //     dl_sr_cuda(image, "ESPCN_x4.pb");
    //     cv::imwrite("result_cuda.jpg", image);
    // }

    return 0;
}




// int main(int argc, char* argv[]){
//     QApplication app(argc, argv);

//     MainWindow m;
//     m.show();

//     return app.exec();
// }