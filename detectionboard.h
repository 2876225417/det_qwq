#ifndef DETECTIONBOARD_H
#define DETECTIONBOARD_H


#include "dbconn.h"
#include "opencv2/videoio.hpp"
#include <QWidget>
#include <QPushButton>
#include <QFileDialog>
#include <QVBoxLayout>
#include <QImage>
#include <memory>
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
#include <qcontainerfwd.h>

#include <QTableWidget>
#include <QMutex>
#include <qmutex.h>
#include <stdexcept>
#include <QComboBox>
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
    QTableWidget*       m_det_result_table;
    QComboBox*          border_color_combobox;
    QComboBox*          font_type_combobox;
    QComboBox*          font_color_combobox;
    QComboBox*          filling_color_combobox;
    QComboBox*          label_position_combobox;

    void                _show_image(const cv::Mat& mat, QLabel* label);
    QStringList         m_labels;
};

enum class video_source_type { file, stream };

class video_infer_thread: public QThread{
    Q_OBJECT
public:
    explicit video_infer_thread( int camera_id = 1
                      , QObject* parent = nullptr
                      )
        : QThread(parent)
        , cap_(new cv::VideoCapture(camera_id)){
            if (!cap_->isOpened()) 
                throw std::runtime_error("Failed to open camera");
        }

    explicit video_infer_thread( const std::string& in_source_file
                      , QObject* parent = nullptr
                      )
        : QThread(parent)
        , cap_(new cv::VideoCapture(in_source_file)){}


    void run() override{
        while(true){
            qDebug() << "Here";
            {
                QMutexLocker locker(&m_mutex);
                if (m_stop_requested || !cap_->isOpened()) break;
            }
            cv::Mat frame;
            (*cap_) >> frame;

            if(frame.empty())
                break;

            cv::Mat infer_result_image = inference_session._infer(frame);

            QStringList labels = inference_session.get_labels();

            detection_result dr;
            for (auto& box: inference_session.get_boxes())
                dr.boxes.append(QRect(box.x, box.y, box.width, box.height));

            dr.class_ids = QVector<int>(inference_session.get_class_ids().begin(), inference_session.get_class_ids().end());
            dr.scores = QVector<float>(inference_session.get_scores().begin(), inference_session.get_scores().end());

            emit setup_labels(labels);
            emit result_ready(dr);
            emit _frame_processed(infer_result_image);
            msleep(10);
        }
    }

    void stop() {
        QMutexLocker locker(&m_mutex);
        m_stop_requested = true;
    }
signals:
    void _frame_processed(const cv::Mat& result);
    void result_ready(const detection_result& res);
    void setup_labels(const QStringList& labels);
    // draw config
    void update_border_color_request(border_color);
    void update_font_type_request(font_type);
    void update_font_color_request(font_color);
    void update_filling_color_request(filling_color);
    void update_label_position_request(label_position);
    // detection config
    void update_score_threshold_request(float);
    void update_nms_1st_request(float);
public slots:
    // draw config
    void handle_border_color_change(border_color bc) {
        QMutexLocker locker(&m_mutex);
        inference_session.set_border_color(bc);
    }
    void handle_font_type_change(font_type ft) {
        QMutexLocker locker(&m_mutex);
        inference_session.set_font_type(ft);
    }
    void handle_font_color_change(font_color fc) {
        QMutexLocker locker(&m_mutex);
        inference_session.set_font_color(fc);
    }
    void handle_filling_color_change(filling_color fic) {
        QMutexLocker locker(&m_mutex);
        inference_session.set_filling_color(fic);
    }
    void handle_label_position_change(label_position lp) {
        QMutexLocker locker(&m_mutex);
        inference_session.set_label_position(lp);
    }
    // detection config
    void handle_score_threshold_change(float threhsold) {
        QMutexLocker locker(&m_mutex);
        inference_session.set_score_threshold(threhsold);
    }
    void handle_nms_1st_threshold_change(float nms_1st) {
        QMutexLocker locker(&m_mutex);
        inference_session.set_nms_1st_threshold(nms_1st);
    }
private:
    // cv::VideoCapture* cap_;
    std::unique_ptr<cv::VideoCapture> cap_;
public: 
    onnxruntime_inference_session inference_session;
    QMutex             m_mutex;
    bool m_stop_requested = false;
};


#endif // DETECTIONBOARD_H
