#pragma once

#include <QMainWindow>
#include <opencv2/opencv.hpp>

#include <QPushButton>
#include <QLabel>
#include <QStatusBar>

QT_BEGIN_NAMESPACE
class QLabel;
class QPushButton;
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);

private slots:
    void loadImage();
    void runCpuVersion();
    void runGpuVersion();

private:
    void showImage(const cv::Mat& mat, QLabel* label);
    void updateResult(const QString& method, double time);
    
    QLabel *originalLabel;
    QLabel *cpuResultLabel;
    QLabel *gpuResultLabel;
    QPushButton *cpuButton;
    QPushButton *gpuButton;
    
    cv::Mat inputImage;
    bool cudaAvailable = false;
};