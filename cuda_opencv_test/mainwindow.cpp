#include "mainwindow.h"
#include <QFileDialog>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QTimer>
#include <QMessageBox>
#include <cuda_runtime.h>
#include <opencv2/cudafilters.hpp>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    // 初始化界面
    originalLabel = new QLabel("Original Image");
    cpuResultLabel = new QLabel("CPU Result");
    gpuResultLabel = new QLabel("GPU Result");
    
    QPushButton *loadButton = new QPushButton("Load Image");
    cpuButton = new QPushButton("Run CPU Version");
    gpuButton = new QPushButton("Run GPU Version");
    
    // 检测CUDA可用性
    cudaAvailable = cv::cuda::getCudaEnabledDeviceCount() > 0;
    gpuButton->setEnabled(cudaAvailable);

    // 布局
    QWidget *centralWidget = new QWidget;
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);
    
    QHBoxLayout *imageLayout = new QHBoxLayout;
    imageLayout->addWidget(originalLabel);
    imageLayout->addWidget(cpuResultLabel);
    imageLayout->addWidget(gpuResultLabel);
    
    QHBoxLayout *buttonLayout = new QHBoxLayout;
    buttonLayout->addWidget(loadButton);
    buttonLayout->addWidget(cpuButton);
    buttonLayout->addWidget(gpuButton);
    
    mainLayout->addLayout(imageLayout);
    mainLayout->addLayout(buttonLayout);
    
    setCentralWidget(centralWidget);
    
    // 连接信号
    connect(loadButton, &QPushButton::clicked, this, &MainWindow::loadImage);
    connect(cpuButton, &QPushButton::clicked, this, &MainWindow::runCpuVersion);
    connect(gpuButton, &QPushButton::clicked, this, &MainWindow::runGpuVersion);
}

void MainWindow::loadImage()
{
    QString path = QFileDialog::getOpenFileName(this, 
        "Open Image", "", "Images (*.png *.jpg)");
    
    if(!path.isEmpty()) {
        inputImage = cv::imread(path.toStdString());
        cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2RGB);
        showImage(inputImage, originalLabel);
    }
}

void MainWindow::runCpuVersion()
{
    if(inputImage.empty()) return;
    
    cv::Mat output;
    const int kernelSize = 15;
    const int runs = 100;
    
    // 预热
    cv::GaussianBlur(inputImage, output, cv::Size(kernelSize, kernelSize), 0);
    
    // 计时
    double t = (double)cv::getTickCount();
    for(int i = 0; i < runs; ++i) {
        cv::GaussianBlur(inputImage, output, 
                       cv::Size(kernelSize, kernelSize), 0);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    
    showImage(output, cpuResultLabel);
    updateResult("CPU", t / runs * 1000);
}

void MainWindow::runGpuVersion()
{
    if(!cudaAvailable || inputImage.empty()) return;
    
    cv::cuda::GpuMat d_src, d_dst;
    cv::Mat output;
    const int kernelSize = 15;
    const int runs = 100;
    
    // 上传数据到GPU
    d_src.upload(inputImage);
    
    // 创建滤波器
    auto filter = cv::cuda::createGaussianFilter(
        CV_8UC3, CV_8UC3, cv::Size(kernelSize, kernelSize), 0);
    
    // 预热
    filter->apply(d_src, d_dst);
    
    // 计时
    double t = (double)cv::getTickCount();
    for(int i = 0; i < runs; ++i) {
        filter->apply(d_src, d_dst);
    }
    cudaDeviceSynchronize(); // 等待GPU完成
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    
    // 下载结果
    d_dst.download(output);
    showImage(output, gpuResultLabel);
    updateResult("GPU", t / runs * 1000);
}

void MainWindow::showImage(const cv::Mat& mat, QLabel* label)
{
    QImage img(mat.data, mat.cols, mat.rows, 
             mat.step, QImage::Format_RGB888);
    label->setPixmap(QPixmap::fromImage(img.scaled(400, 300, 
        Qt::KeepAspectRatio)));
}

void MainWindow::updateResult(const QString& method, double time)
{
    QString msg = QString("%1 Time: %2 ms").arg(method).arg(time, 0, 'f', 2);
    statusBar()->showMessage(msg);
}