#include "detectionboard.h"
#include "dbconn.h"
#include "qt_utils.hpp"
#include <QFormLayout>

#include <QBoxLayout>
#include <qboxlayout.h>
#include <qcheckbox.h>
#include <qcombobox.h>
#include <qdatetime.h>
#include <qformlayout.h>
#include <qgroupbox.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qlogging.h>
#include <qoverload.h>
#include <qpushbutton.h>
#include <qspinbox.h>
#include <qsqlquery.h>
#include <qvariant.h>
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
    
    QHBoxLayout* detection_board_layout = new QHBoxLayout();

    QHBoxLayout* detection_board_layout_wrapper = new QHBoxLayout();
    QGroupBox*   detection_board = new QGroupBox();
    // config panel
    QVBoxLayout* config_panel_layout_wrapper = new QVBoxLayout();
    QGroupBox* config_panel = new QGroupBox("Config");

    QVBoxLayout* model_config_layout_wrapper = new QVBoxLayout();
    QGroupBox* model_config_layout = new QGroupBox("Model config");
    QHBoxLayout* model_select_layout_wrapper = new QHBoxLayout();
    QLabel* model_select_id_label = new QLabel("Model id: ");
    QComboBox* model_select_combobox = new QComboBox();
    QPushButton* model_select_button = new QPushButton("Refresh");
    QFormLayout* model_detail_form_wrapper = new QFormLayout();
    QGroupBox* model_detail_form = new QGroupBox();

    QLabel* model_id_detail = new QLabel();
    QLabel* model_path_detail = new QLabel();
    QLabel* model_time_detail = new QLabel();
    QLabel* model_params_detail = new QLabel();
    QLabel* model_mAP50_detail = new QLabel();
    QLabel* model_recall_detail = new QLabel();
    QLabel* model_layers_detail = new QLabel();
    QLabel* model_gradients_detail = new QLabel();
    QLabel* model_precision_detail = new QLabel();

    model_detail_form->setLayout(model_detail_form_wrapper);

    model_detail_form_wrapper->addRow("Model id:", model_id_detail);
    model_detail_form_wrapper->addRow("Save Path:", model_path_detail);
    model_detail_form_wrapper->addRow("Training Time:", model_time_detail);
    model_detail_form_wrapper->addRow("Pramas:", model_params_detail);
    model_detail_form_wrapper->addRow("mAP50:", model_mAP50_detail);
    model_detail_form_wrapper->addRow("Recall:", model_recall_detail);
    model_detail_form_wrapper->addRow("Layers: ", model_layers_detail);
    model_detail_form_wrapper->addRow("Gradients: ", model_gradients_detail);
    model_detail_form_wrapper->addRow("Precision: ", model_precision_detail);

    auto update_model_list = [=]() {
        model_select_combobox->clear();
        auto models = dbConn::instance().get_all_models();
        for (const auto& model: models) {
            QString display_text = QString("%1").arg(model.train_id.left(8));
            model_select_combobox->addItem(display_text, QVariant(model.train_id));
        }
    };

    auto update_model_details = [=](int index) {
        model_id_detail->clear();
        model_path_detail->clear();
        model_time_detail->clear();
        model_params_detail->clear();
        model_mAP50_detail->clear();
        model_recall_detail->clear();

        if (index < 0) return;

        QVariant selected_model_id = model_select_combobox->itemData(index);
        qDebug() << "selected_id: " << selected_model_id;
        if (!selected_model_id.isValid()) {
            model_id_detail->setText("Invalid model option");
            return;
        }

        QSqlQuery query(dbConn::instance().getDatabase());
        query.prepare(
            "SELECT save_path, timestamp, params, mAP50, recall "
            "FROM training_records "
            "WHERE train_id = ?"
        );

        query.addBindValue(selected_model_id.toString());

        if (!query.exec()) {
            qCritical() << "Query failed: " << query.lastQuery().toStdString();
            model_id_detail->setText("Database error");
            return;
        }

        if (query.next()) {
            model_id_detail->setText(selected_model_id.toString());
            model_path_detail->setText(query.value("save_path").toString());

            QDateTime model_gen_date = query.value("timestamp").toDateTime().toLocalTime();
            model_time_detail->setText(model_gen_date.toString("yyyy-MM-dd HH:mm"));

            qint64 params = query.value("params").toLongLong();
            model_params_detail->setText(QLocale().toString(params));

            double map50 = query.value("mAP50").toDouble();
            model_mAP50_detail->setText(QString::number(map50, 'f', map50 < 0.0001 ? 6 : 4));
            
            double recall = query.value("recall").toDouble();
            model_recall_detail->setText(QString("%1%").arg(recall * 100, 0, 'f', 2));
        } else model_id_detail->setText("Not found the corresponding model");

        // 设置 识别时使用的模型路径

    };

    connect( model_select_combobox
        , QOverload<int>::of(&QComboBox::currentIndexChanged)
        , this
        , update_model_details);

    model_select_layout_wrapper->addWidget(model_select_id_label);
    model_select_layout_wrapper->addWidget(model_select_combobox);
    model_select_layout_wrapper->addWidget(model_select_button);
    model_config_layout_wrapper->addLayout(model_select_layout_wrapper);
    model_config_layout_wrapper->addWidget(model_detail_form);

    model_config_layout->setLayout(model_config_layout_wrapper);

    QVBoxLayout* detection_config_layout_wrapper = new QVBoxLayout();
    QGroupBox* detection_config_layout = new QGroupBox("Detection Config");

    QHBoxLayout* enable_cuda_check_layout = new QHBoxLayout();
    QLabel* cuda_check_label = new QLabel("Enable CUDA ");
    QCheckBox* cuda_checkbox = new QCheckBox();
    enable_cuda_check_layout->addWidget(cuda_check_label);
    enable_cuda_check_layout->addWidget(cuda_checkbox);

    QHBoxLayout* adjust_score_threshold_layout = new QHBoxLayout();
    QLabel* score_threshold_label = new QLabel("Threshold");
    QSpinBox* score_threshold_adjuster = new QSpinBox();
    adjust_score_threshold_layout->addWidget(score_threshold_label);
    adjust_score_threshold_layout->addWidget(score_threshold_adjuster);

    QHBoxLayout* adjust_nms_layout = new QHBoxLayout();
    QHBoxLayout* adjust_nms_1st_layout = new QHBoxLayout();
    QLabel* adjust_nms_1st_label = new QLabel("NMS_1");
    QSpinBox* nms_1st_adjuster = new QSpinBox();
    adjust_nms_1st_layout->addWidget(adjust_nms_1st_label);
    adjust_nms_1st_layout->addWidget(nms_1st_adjuster);

    QHBoxLayout* adjust_nms_2nd_layout = new QHBoxLayout();
    QLabel* adjust_nms_2nd_label = new QLabel("NMS_2");
    QSpinBox* nms_2nd_adjuster = new QSpinBox();
    adjust_nms_2nd_layout->addWidget(adjust_nms_2nd_label);
    adjust_nms_2nd_layout->addWidget(nms_2nd_adjuster);

    adjust_nms_layout->addLayout(adjust_nms_1st_layout);
    adjust_nms_layout->addLayout(adjust_nms_2nd_layout);

    QHBoxLayout* enable_cpu_mem_arena_layout = new QHBoxLayout();
    
    QHBoxLayout* set_intra_op_num_threads = new QHBoxLayout();
    QComboBox* intra_op_num_threads_adjuster = new QComboBox();

    QHBoxLayout* select_class_file_layout = new QHBoxLayout();

    QHBoxLayout* set_graph_optimization_level_layout = new QHBoxLayout();


    detection_config_layout_wrapper->addLayout(enable_cuda_check_layout);
    detection_config_layout_wrapper->addLayout(adjust_score_threshold_layout);
    detection_config_layout_wrapper->addLayout(adjust_nms_layout);
    detection_config_layout->setLayout(detection_config_layout_wrapper);


    QVBoxLayout* display_config_layout_wrapper = new QVBoxLayout();
    QGroupBox* display_config_layout = new QGroupBox("Display Config");


    display_config_layout->setLayout(display_config_layout_wrapper);

    config_panel_layout_wrapper->addWidget(model_config_layout);
    config_panel_layout_wrapper->addWidget(detection_config_layout);
    config_panel_layout_wrapper->addWidget(display_config_layout);

    connect (model_select_button, &QPushButton::clicked, update_model_list);

    config_panel->setLayout(config_panel_layout_wrapper);


    // ----------------- //
    QWidget* main_group = new QWidget();
    main_group->setStyleSheet("background-color: #333;"
                              "border-radius: 5px;"
                              "padding: 10px;");


    QVBoxLayout* sub_layout = new QVBoxLayout();
    /*-------------------sub_layout-------------------*/
    QGroupBox* src_group = new QGroupBox("From Image/Video");
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


    // detection panel
    QVBoxLayout* detection_panel_layout_wrapper=  new QVBoxLayout();
    QGroupBox* detection_panel = new QGroupBox("Detection");

    QHBoxLayout* detection_operation_panel_layout_wrapper = new QHBoxLayout();
    QGroupBox* detection_operation_panel = new QGroupBox();

    detection_operation_panel->setLayout(detection_operation_panel_layout_wrapper);

    QHBoxLayout* detection_result_panel_layout_wrapper = new QHBoxLayout();
    QGroupBox* detection_result_panel = new QGroupBox();

    detection_result_panel_layout_wrapper->addLayout(sub_layout);

    detection_result_panel->setLayout(detection_result_panel_layout_wrapper);

    detection_panel_layout_wrapper->addWidget(detection_operation_panel,1);
    detection_panel_layout_wrapper->addWidget(detection_result_panel, 4);

    detection_panel->setLayout(detection_panel_layout_wrapper);

    detection_board_layout_wrapper->addWidget(config_panel, 2);
    detection_board_layout_wrapper->addWidget(detection_panel, 7);
    detection_board->setLayout(detection_board_layout_wrapper);

    cam_group->setLayout(camera_stream_layout);

    sub_layout->addWidget(src_group);
    sub_layout->addWidget(cam_group);

    main_group->setLayout(sub_layout);


    _push_button_stop_video->setEnabled(false);
    layout_1->addWidget(main_group);


    detection_board_layout->addWidget(detection_board);
    setLayout(detection_board_layout);

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


