#ifndef DETECTIONBOARD_H
#define DETECTIONBOARD_H


#include <QWidget>
#include <QPushButton>
#include <QFileDialog>
#include <QVBoxLayout>
#include <QImage>
#include <opencv2/opencv.hpp>
#include <QPixmap>
#include <QLabel>
#include <QMessageBox>
#include <QDebug>
#include <QThread>
#include <QPainter>
#include <QHBoxLayout>
#include <QImageReader>
#include <QTimer>
#include <chrono>
#include <QGroupBox>
#include <QTimer>

#include "onnxruntime_inference_session.h"

class video_infer_thread;

class detectionBoard: public QWidget
{
    Q_OBJECT
public:
    explicit detectionBoard(QWidget* parent = nullptr);
    ~detectionBoard();

private slots:
    void _on_load_source_clicked();
    void _on_infer_source_clicked();

    void _on_start_video_clicked();
    void _on_stop_video_clicked();

    void _on_frame_update();
    void _show_infer_result(const cv::Mat& infer_result_image);

    void _show_src_camera_result(const cv::Mat& src_camera_frame);
    void _show_ifd_camera_result(const cv::Mat& ifd_camera_frame);



private:
    QLabel*             _label;
    QLabel*             _infered_label;
    QPushButton*        _push_button_load_image;
    QPushButton*        _push_button_infer_image;

    QLabel*             _camera_label;
    QLabel*             _ifd_camera_label;
    QPushButton*        _push_button_start_video;
    QPushButton*        _push_button_stop_video;

    std::string         _source_path;
    cv::VideoCapture    _cap;

    QTimer*             _video_timer;
    video_infer_thread* _video_infer_thread;
    video_infer_thread* _video_infer_thread_from_source;

    void                _show_image(const cv::Mat& mat, QLabel* label);
};

enum class video_source_type { file, stream };

class video_infer_thread: public QThread{
    Q_OBJECT
public:
    explicit video_infer_thread( cv::VideoCapture* cap
                      , QObject* parent = nullptr
                      )
        : QThread(parent)
        , cap_(cap){}

    explicit video_infer_thread( const std::string& in_source_file
                      , QObject* parent = nullptr
                      )
        : QThread(parent)
        , cap_(new cv::VideoCapture(in_source_file)){}

    void run() override{
        while(cap_->isOpened()){
            cv::Mat frame;
            (*cap_) >> frame;

            if(frame.empty())
                break;

            onnxruntime_inference_session inference_session;
            cv::Mat infer_result_image = inference_session._infer(frame);

            emit _frame_processed(infer_result_image);
            msleep(30);
        }
    }
signals:
    void _frame_processed(const cv::Mat& result);

private:
    cv::VideoCapture* cap_;
};


#endif // DETECTIONBOARD_H
