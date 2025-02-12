#include "detectionboard.h"
#include "qt_utils.hpp"
#include <QFormLayout>

#include <QBoxLayout>
detectionBoard::detectionBoard(QWidget* parent)
    : QWidget(parent)
    , _label(new QLabel())
    , _infered_label(new QLabel())
    , _push_button_load_image(new QPushButton("Load Image", this))
    , _push_button_infer_image(new QPushButton("Infer", this))
    , _camera_label(new QLabel(this))
    , _ifd_camera_label(new QLabel(this))
    , _push_button_start_video(new QPushButton("Start Video", this))
    , _push_button_stop_video(new QPushButton("Stop Video", this))
    , _video_timer(new QTimer(this)){

    // #1 detection main layout
    QHBoxLayout* layout_1 = new QHBoxLayout();

    QWidget* main_group = new QWidget();
    main_group->setStyleSheet("background-color: #333;"
                              "border-radius: 5px;"
                              "padding: 10px;");


    QVBoxLayout* sub_layout = new QVBoxLayout();
    /*-------------------sub_layout-------------------*/
    QGroupBox* src_group = new QGroupBox("From Source");
    src_group->setStyleSheet("background-color: #444;");
    QHBoxLayout* src_selection_layout = new QHBoxLayout();
    QGroupBox* selection_buttons_group = new QGroupBox();
    selection_buttons_group->setStyleSheet("background-color: #333;");
    QVBoxLayout* selection_buttons_layout = new QVBoxLayout();

    selection_buttons_layout->addWidget(_push_button_load_image);
    selection_buttons_layout->addWidget(_push_button_infer_image);

    /*--------------------src_group----------------------*/
    QGroupBox* src_ifd_group = new QGroupBox("Detection Area");
    src_ifd_group->setStyleSheet("background-color: #555");
    QHBoxLayout* src_ifd_layout = new QHBoxLayout();

    QWidget* src_img = new QWidget();
    src_img->setStyleSheet("background-color: #777;");
    QVBoxLayout* src_img_layout = new QVBoxLayout();
    src_img_layout->addWidget(_label);
    src_img->setLayout(src_img_layout);

    QWidget* ifd_img = new QWidget();
    ifd_img->setStyleSheet("background-color: #666;");
    QVBoxLayout* ifd_img_layout = new QVBoxLayout();
    ifd_img_layout->addWidget(_infered_label);
    ifd_img->setLayout(ifd_img_layout);

    src_ifd_layout->addWidget(src_img);
    src_ifd_layout->addWidget(ifd_img);
    selection_buttons_group->setLayout(selection_buttons_layout);
    src_ifd_group->setLayout(src_ifd_layout);

    src_selection_layout->addWidget(selection_buttons_group, 1);
    src_selection_layout->addWidget(src_ifd_group, 8);
    src_group->setLayout(src_selection_layout);
    /*-------------------cam_group----------------------*/
    QGroupBox* cam_group = new QGroupBox("From Camera");
    QHBoxLayout* camera_stream_layout = new QHBoxLayout();
    QGroupBox* cam_operations_group = new QGroupBox();
    QVBoxLayout* cam_operations_layout = new QVBoxLayout();
    cam_operations_layout->addWidget(_push_button_start_video);
    cam_operations_layout->addWidget(_push_button_stop_video);
    cam_operations_group->setLayout(cam_operations_layout);

    QGroupBox* cam_streaming_group = new QGroupBox();
    QHBoxLayout* cam_streaming_layout = new QHBoxLayout();

    QWidget* src_cam_stream = new QWidget();
    QVBoxLayout* src_cam_stream_layout = new QVBoxLayout();
    src_cam_stream_layout->addWidget(_camera_label);
    src_cam_stream->setLayout(src_cam_stream_layout);

    QWidget* src_cam_ifd_stream =  new QWidget();
    QVBoxLayout* src_cam_ifd_stream_layout = new QVBoxLayout();
    src_cam_ifd_stream_layout->addWidget(_ifd_camera_label);
    src_cam_ifd_stream->setLayout(src_cam_ifd_stream_layout);


    cam_streaming_layout->addWidget(src_cam_stream);
    cam_streaming_layout->addWidget(src_cam_ifd_stream);

    cam_streaming_group->setLayout(cam_streaming_layout);

    camera_stream_layout->addWidget(cam_operations_group, 1);
    camera_stream_layout->addWidget(cam_streaming_group, 8);
    /*-------------------------------------------------*/

    cam_group->setLayout(camera_stream_layout);

    sub_layout->addWidget(src_group);
    sub_layout->addWidget(cam_group);

    main_group->setLayout(sub_layout);
    _push_button_stop_video->setEnabled(false);


    layout_1->addWidget(main_group);
    setLayout(layout_1);

    connect(_push_button_load_image
           , &QPushButton::clicked
           , this
           , &detectionBoard::_on_load_source_clicked
           );
    connect(_push_button_infer_image
           , &QPushButton::clicked
           , this
           , &detectionBoard::_on_infer_source_clicked
           );
    connect(_push_button_start_video
           , &QPushButton::clicked
           , this
           , &detectionBoard::_on_start_video_clicked
           );
    connect(_push_button_stop_video
           , &QPushButton::clicked
           , this
           , &detectionBoard::_on_stop_video_clicked
           );
    connect(_video_timer
           , &QTimer::timeout
           , this
           , &detectionBoard::_on_frame_update
           );
}

detectionBoard::~detectionBoard() {}


void detectionBoard::_on_load_source_clicked(){
    QString file_name = QFileDialog::getOpenFileName( this
                                                    , tr("Open Source")
                                                    , ""
                                                    , tr("Image Files (*.png *.jpg *.bmp *.jpeg);;Video Files (*.mp4 *.avi *.mov *.mkv"));
    if(file_name.isEmpty()){
        qDebug() << "No file selected.";
        return;
    }
    _source_path = file_name.toStdString();
    _show_image(cv::imread(_source_path), _label);
}


void detectionBoard::_on_infer_source_clicked(){
    if(_source_path.empty())
        return;

    QFileInfo file_info(QString::fromStdString(_source_path));
    QString file_extension = file_info.suffix().toLower();

    std::cout << "start infer\n";

    if(file_extension == "mp4"
    || file_extension == "avi"
    || file_extension == "mov"
    || file_extension == "mkv"){
        _video_infer_thread_from_source = new video_infer_thread( _source_path
                                                                , this
                                                                );
        connect( _video_infer_thread_from_source
               , &video_infer_thread::_frame_processed
               , this
               , &detectionBoard::_show_infer_result
               );

        _video_infer_thread_from_source->start();
    } else if(file_extension == "png"
           || file_extension == "jpg"
           || file_extension == "jpeg"
           || file_extension == "bmp"
              ){
        onnxruntime_inference_session inference_session;
        cv::Mat infer_result_image = inference_session._infer(cv::imread(_source_path));
        std::cout << "infer over\n";
        _show_image(infer_result_image, _infered_label);
    } else qDebug() << "No matching file type!";
}

void detectionBoard::_on_start_video_clicked(){
    _cap.open(0);
    cv::Mat src_from_camera;
    _cap >> src_from_camera;
    if(!_cap.isOpened()){
        std::cerr << "Error: Unable to open video stream.\n";
        return;
    }


    // QTimer* timer = new QTimer(this);

    // connect(timer, &QTimer::timeout, this, &detectionBoard::_show_src_camera_result);

    // timer->start(1000);

    _video_infer_thread = new video_infer_thread(&_cap, this);

    connect( _video_infer_thread
           , &video_infer_thread::_frame_processed
           , this
           , &detectionBoard::_show_ifd_camera_result
           );

    _video_infer_thread->start();

    _push_button_start_video->setEnabled(false);
    _push_button_stop_video->setEnabled(true);

}

void detectionBoard::_on_stop_video_clicked(){
    _cap.release();
    _video_timer->stop();

    _push_button_start_video->setEnabled(true);
    _push_button_stop_video->setEnabled(false);
}

void detectionBoard::_show_infer_result(const cv::Mat& infer_result_image){
    _show_image(infer_result_image, _infered_label);
}

void detectionBoard::_show_ifd_camera_result(const cv::Mat& ifd_camera_frame){
    _show_image(ifd_camera_frame, _ifd_camera_label);
}

void detectionBoard::_show_src_camera_result(const cv::Mat& src_camera_frame) {
    _show_image(src_camera_frame, _camera_label);
}

void detectionBoard::_on_frame_update(){
    cv::Mat frame_;
    _cap >> frame_;

    if(frame_.empty())
        return;

    onnxruntime_inference_session inference_session;
    cv::Mat infer_result_image = inference_session._infer(frame_);

    _show_image(infer_result_image, _infered_label);
}

void detectionBoard::_show_image(const cv::Mat& mat, QLabel* label){
    if(mat.empty()) return;
    auto start = std::chrono::high_resolution_clock::now();

    std::optional<QImage> img = qt_utils::mat_to_image(mat);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    qDebug() << "Image conversion took: " << duration.count() << "seconds\n";

    if(!img){
        qDebug() << "Failed to conver cv::Mat to QImage";
        return;
    }

    QPixmap tmp = QPixmap::fromImage(*img);

    QPixmap scaled_pixmap = tmp.scaled(tmp.width() / 2
                                      , tmp.height() / 2
                                      , Qt::KeepAspectRatio
                                      , Qt::SmoothTransformation
                                      );

    label->setPixmap(scaled_pixmap);
    label->setScaledContents(false);
}


