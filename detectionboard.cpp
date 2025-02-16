#include "detectionboard.h"
#include "dbconn.h"
#include "onnxruntime_inference_session.h"
#include "qt_utils.hpp"
#include <QFormLayout>

#include <QBoxLayout>
#include <qabstractitemview.h>
#include <qboxlayout.h>
#include <qcheckbox.h>
#include <qcombobox.h>
#include <qcontainerfwd.h>
#include <qdatetime.h>
#include <qformlayout.h>
#include <qgroupbox.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qlogging.h>
#include <qnamespace.h>
#include <qoverload.h>
#include <qpushbutton.h>
#include <qsizepolicy.h>
#include <qspinbox.h>
#include <qsqlquery.h>
#include <qtablewidget.h>
#include <qvariant.h>


QComboBox* workers_combobox_with_concurrency(QWidget* parent = nullptr);


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
    QLabel* model_mAP5095_detail = new QLabel();

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
    model_detail_form_wrapper->addRow("mAP5095", model_mAP5095_detail);

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
        model_mAP5095_detail->clear();
        model_recall_detail->clear();
        model_layers_detail->clear();
        model_gradients_detail->clear();
        model_precision_detail->clear();

        if (index < 0) return;

        QVariant selected_model_id = model_select_combobox->itemData(index);
        qDebug() << "selected_id: " << selected_model_id;
        if (!selected_model_id.isValid()) {
            model_id_detail->setText("Invalid model option");
            return;
        }

        QSqlQuery query(dbConn::instance().getDatabase());
        query.prepare(
            "SELECT save_path, timestamp, params, mAP50, mAP5095, recall, layers, precision, gradients "
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
         
            double map5095 = query.value("mAP5095").toDouble();
            model_mAP5095_detail->setText(QString::number(map5095, 'f', map5095 < 0.0001 ? 6 : 4));
              
            double recall = query.value("recall").toDouble();
            model_recall_detail->setText(QString("%1%").arg(recall * 100, 0, 'f', 2));
        
            int layers = query.value("layers").toInt();
            model_layers_detail->setText(QString::number(layers));

            double precision = query.value("precision").toDouble();
            model_precision_detail->setText(QString::number(precision, 'f', 4));

            int gradients = query.value("gradients").toInt();
            model_gradients_detail->setText(QString::number(gradients));
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
    
    QHBoxLayout* select_detection_type_layout = new QHBoxLayout();
    QLabel* detection_type_label = new QLabel("Det Type");
    QComboBox* detection_type_combobox = new QComboBox();
    detection_type_combobox->addItem("Detect");
    detection_type_combobox->addItem("Segment");
    detection_type_combobox->addItem("Classify");
    detection_type_combobox->addItem("Pose");
    select_detection_type_layout->addWidget(detection_type_label);
    select_detection_type_layout->addWidget(detection_type_combobox);
    
    QHBoxLayout* enable_cuda_check_layout = new QHBoxLayout();
    QLabel* cuda_check_label = new QLabel("Enable CUDA ");
    QCheckBox* cuda_checkbox = new QCheckBox();
    enable_cuda_check_layout->addWidget(cuda_check_label);
    enable_cuda_check_layout->addWidget(cuda_checkbox);

    QHBoxLayout* adjust_score_threshold_layout = new QHBoxLayout();
    QLabel* score_threshold_label = new QLabel("Threshold");
    QSpinBox* score_threshold_adjuster = new QSpinBox();
    score_threshold_adjuster->setRange(1, 100);
    score_threshold_adjuster->setValue(10);
    score_threshold_adjuster->setSingleStep(5);
    adjust_score_threshold_layout->addWidget(score_threshold_label);
    adjust_score_threshold_layout->addWidget(score_threshold_adjuster);

    connect ( score_threshold_adjuster
            , QOverload<int>::of(&QSpinBox::valueChanged)
            , this
            , [this](int value) {
                float score_threshold =  value / 100.f;
                emit _video_infer_thread->update_score_threshold_request(score_threshold);
            });

    QHBoxLayout* adjust_nms_layout = new QHBoxLayout();
    QHBoxLayout* adjust_nms_1st_layout = new QHBoxLayout();
    QLabel* adjust_nms_1st_label = new QLabel("NMS_1");
    QSpinBox* nms_1st_adjuster = new QSpinBox();
    nms_1st_adjuster->setRange(0, 100);
    nms_1st_adjuster->setValue(50);
    nms_1st_adjuster->setSingleStep(5);
    adjust_nms_1st_layout->addWidget(adjust_nms_1st_label);
    adjust_nms_1st_layout->addWidget(nms_1st_adjuster);

    connect ( nms_1st_adjuster
            , QOverload<int>::of(&QSpinBox::valueChanged)
            , this
            , [this](int value) {
                float nms_1st = value / 100.f;
                emit _video_infer_thread->update_nms_1st_request(nms_1st);
            });

    QHBoxLayout* adjust_nms_2nd_layout = new QHBoxLayout();
    QLabel* adjust_nms_2nd_label = new QLabel("NMS_2");
    QSpinBox* nms_2nd_adjuster = new QSpinBox();
    nms_2nd_adjuster->setRange(0, 100);
    nms_2nd_adjuster->setValue(50);
    nms_2nd_adjuster->setSingleStep(5);
    adjust_nms_2nd_layout->addWidget(adjust_nms_2nd_label);
    adjust_nms_2nd_layout->addWidget(nms_2nd_adjuster);

    adjust_nms_layout->addLayout(adjust_nms_1st_layout);
    adjust_nms_layout->addLayout(adjust_nms_2nd_layout);

    QHBoxLayout* enable_cpu_mem_arena_layout = new QHBoxLayout();
    QLabel* check_cpu_mem_arena_label = new QLabel("Memory Arena");
    QCheckBox* cpu_mem_arena_checkbox = new QCheckBox();
    enable_cpu_mem_arena_layout->addWidget(check_cpu_mem_arena_label);
    enable_cpu_mem_arena_layout->addWidget(cpu_mem_arena_checkbox);
    
    QHBoxLayout* set_intra_op_num_threads_layout = new QHBoxLayout();
    QLabel* intra_op_num_label = new QLabel("Intra Op");
    QComboBox* intra_op_num_threads_adjuster = workers_combobox_with_concurrency();
    set_intra_op_num_threads_layout->addWidget(intra_op_num_label);
    set_intra_op_num_threads_layout->addWidget(intra_op_num_threads_adjuster);

    QHBoxLayout* select_class_file_layout = new QHBoxLayout();
    QLabel* class_file_label = new QLabel("Class File");
    QPushButton* select_class_file_button = new QPushButton("Select");
    select_class_file_layout->addWidget(class_file_label);
    select_class_file_layout->addWidget(select_class_file_button);
    
    QHBoxLayout* set_graph_optimization_level_layout = new QHBoxLayout();
    QLabel* graph_optimization_level_label = new QLabel("Graph Optimization");
    QComboBox* select_graph_optimization_level_combobox = new QComboBox();
    select_graph_optimization_level_combobox->addItem("Disable All");
    select_graph_optimization_level_combobox->addItem("Enable Basic");
    select_graph_optimization_level_combobox->addItem("Enable Extended");
    select_graph_optimization_level_combobox->addItem("Enable All");
    set_graph_optimization_level_layout->addWidget(graph_optimization_level_label);
    set_graph_optimization_level_layout->addWidget(select_graph_optimization_level_combobox);

    detection_config_layout_wrapper->addLayout(select_detection_type_layout);
    detection_config_layout_wrapper->addLayout(enable_cuda_check_layout);
    detection_config_layout_wrapper->addLayout(adjust_score_threshold_layout);
    detection_config_layout_wrapper->addLayout(adjust_nms_layout);
    detection_config_layout_wrapper->addLayout(enable_cpu_mem_arena_layout);
    detection_config_layout_wrapper->addLayout(set_intra_op_num_threads_layout);
    detection_config_layout_wrapper->addLayout(select_class_file_layout);
    detection_config_layout_wrapper->addLayout(set_graph_optimization_level_layout);
    detection_config_layout->setLayout(detection_config_layout_wrapper);


    QVBoxLayout* display_config_layout_wrapper = new QVBoxLayout();
    QGroupBox* display_config_layout = new QGroupBox("Display Config");
     
    QHBoxLayout* adjust_detection_border_color_wraper = new QHBoxLayout();
    QLabel* detection_border_color_label = new QLabel("border Color");
    border_color_combobox = new QComboBox();
    border_color_combobox->addItem("red", QVariant::fromValue(border_color::red));
    border_color_combobox->addItem("green", QVariant::fromValue(border_color::green));
    border_color_combobox->addItem("blue", QVariant::fromValue(border_color::blue));
    border_color_combobox->addItem("white", QVariant::fromValue(border_color::white));
    border_color_combobox->addItem("black", QVariant::fromValue(border_color::black));
    border_color_combobox->addItem("cyan", QVariant::fromValue(border_color::cyan));
    adjust_detection_border_color_wraper->addWidget(detection_border_color_label);
    adjust_detection_border_color_wraper->addWidget(border_color_combobox);

    connect (border_color_combobox
            , QOverload<int>::of(&QComboBox::currentIndexChanged)
            , this, [this](int index) {
                border_color bc = border_color_combobox->itemData(index).value<border_color>();
                emit _video_infer_thread->update_border_color_request(bc);
            });

    QHBoxLayout* adjust_detection_font_type_wraper = new QHBoxLayout();
    QLabel* detection_font_type_label = new QLabel("Font Type");
    font_type_combobox = new QComboBox();
    font_type_combobox->addItem("simplex", QVariant::fromValue(font_type::simplex));
    font_type_combobox->addItem("plain", QVariant::fromValue(font_type::plain));
    font_type_combobox->addItem("duplex", QVariant::fromValue(font_type::duplex));
    font_type_combobox->addItem("complex", QVariant::fromValue(font_type::complex));
    font_type_combobox->addItem("triplex", QVariant::fromValue(font_type::triplex));
    font_type_combobox->addItem("complex_small", QVariant::fromValue(font_type::complex_small));
    adjust_detection_font_type_wraper->addWidget(detection_font_type_label);
    adjust_detection_font_type_wraper->addWidget(font_type_combobox);

    connect ( font_type_combobox
            , QOverload<int>::of(&QComboBox::currentIndexChanged)
            , this
            , [this](int index) {
                font_type ft = font_type_combobox->itemData(index).value<font_type>();
                emit _video_infer_thread->update_font_type_request(ft);
            });
    
    QHBoxLayout* adjust_detection_font_color_wraper = new QHBoxLayout();
    QLabel* detection_font_color_label = new QLabel("Font Color");
    font_color_combobox = new QComboBox();
    font_color_combobox->addItem("high contrast", QVariant::fromValue(font_color::high_contrast));
    font_color_combobox->addItem("pure white", QVariant::fromValue(font_color::pure_white));
    font_color_combobox->addItem("neon green", QVariant::fromValue(font_color::neon_green));
    font_color_combobox->addItem("warning yellow", QVariant::fromValue(font_color::warning_yellow));
    font_color_combobox->addItem("signal red", QVariant::fromValue(font_color::signal_red));
    adjust_detection_font_color_wraper->addWidget(detection_font_color_label);
    adjust_detection_font_color_wraper->addWidget(font_color_combobox);

    connect ( font_color_combobox
            , QOverload<int>::of(&QComboBox::currentIndexChanged)
            , this
            , [this](int index) {
                font_color fc = font_color_combobox->itemData(index).value<font_color>();
                emit _video_infer_thread->update_font_color_request(fc);
            });

    QHBoxLayout* select_filling_color_wrapper = new QHBoxLayout();
    QLabel* filling_color_label = new QLabel("Filling Color");
    filling_color_combobox = new QComboBox();
    filling_color_combobox->addItem("none", QVariant::fromValue(filling_color::None));
    filling_color_combobox->addItem("dark overlay", QVariant::fromValue(filling_color::dark_overlay));
    filling_color_combobox->addItem("light overlay", QVariant::fromValue(filling_color::light_overlay));
    filling_color_combobox->addItem("danger highlight", QVariant::fromValue(filling_color::danger_highlight));
    filling_color_combobox->addItem("info highlight", QVariant::fromValue(filling_color::info_highlight));
    select_filling_color_wrapper->addWidget(filling_color_label);
    select_filling_color_wrapper->addWidget(filling_color_combobox);

    connect ( filling_color_combobox
            , QOverload<int>::of(&QComboBox::currentIndexChanged)
            , this
            , [this](int index) {
                filling_color fic = filling_color_combobox->itemData(index).value<filling_color>();
                emit _video_infer_thread->update_filling_color_request(fic);
            });

    QHBoxLayout* select_label_position_wrapper = new QHBoxLayout();
    QLabel* label_positon_label = new QLabel("Label Position");
    label_position_combobox = new QComboBox();
    label_position_combobox->addItem("Top Left", QVariant::fromValue(label_position::top_left));
    label_position_combobox->addItem("Top right", QVariant::fromValue(label_position::top_right));
    label_position_combobox->addItem("Bottom Left", QVariant::fromValue(label_position::bottom_left));
    label_position_combobox->addItem("Bottom Right", QVariant::fromValue(label_position::bottom_right));
    label_position_combobox->addItem("Center", QVariant::fromValue(label_position::center));
    select_label_position_wrapper->addWidget(label_positon_label);
    select_label_position_wrapper->addWidget(label_position_combobox);

    connect ( label_position_combobox
            , QOverload<int>::of(&QComboBox::currentIndexChanged)
            , this
            , [this](int index) {
                label_position lp = label_position_combobox->itemData(index).value<label_position>();
                emit _video_infer_thread->update_label_position_request(lp);
            });
    
    display_config_layout_wrapper->addLayout(adjust_detection_border_color_wraper);
    display_config_layout_wrapper->addLayout(adjust_detection_font_type_wraper);
    display_config_layout_wrapper->addLayout(adjust_detection_font_color_wraper);
    display_config_layout_wrapper->addLayout(select_filling_color_wrapper);
    display_config_layout_wrapper->addLayout(select_label_position_wrapper);
    
    display_config_layout->setLayout(display_config_layout_wrapper);

    config_panel_layout_wrapper->addWidget(model_config_layout, 3);
    config_panel_layout_wrapper->addWidget(detection_config_layout, 4);
    config_panel_layout_wrapper->addWidget(display_config_layout, 2);

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


    //cam_streaming_layout->addWidget(src_cam_stream);
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

    m_det_result_table = new QTableWidget();
    m_det_result_table->setColumnCount(4);

    m_det_result_table->setHorizontalHeaderLabels({"Class", "Confidence", "Position", "size"});
    m_det_result_table->setEditTriggers(QAbstractItemView::NoEditTriggers);

    detection_operation_panel_layout_wrapper->addWidget(m_det_result_table);
    detection_operation_panel->setLayout(detection_operation_panel_layout_wrapper);

    QHBoxLayout* detection_result_panel_layout_wrapper = new QHBoxLayout();
    QGroupBox* detection_result_panel = new QGroupBox();

    detection_result_panel_layout_wrapper->addLayout(sub_layout);

    detection_result_panel->setLayout(detection_result_panel_layout_wrapper);

    detection_panel_layout_wrapper->addWidget(detection_result_panel,4);
    detection_panel_layout_wrapper->addWidget(detection_operation_panel, 1);

    detection_panel->setLayout(detection_panel_layout_wrapper);

    detection_board_layout_wrapper->addWidget(config_panel, 2);
    detection_board_layout_wrapper->addWidget(detection_panel, 7);
    detection_board->setLayout(detection_board_layout_wrapper);

    cam_group->setLayout(camera_stream_layout);

    sub_layout->addWidget(src_group, 1);
    sub_layout->addWidget(cam_group, 1);

    //main_group->setLayout(sub_layout);


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
    // _cap.open(0);
    // cv::Mat src_from_camera;
    // _cap >> src_from_camera;
    // if(!_cap.isOpened()){
    //     std::cerr << "Error: Unable to open video stream.\n";
    //     return;
    // }


    // QTimer* timer = new QTimer(this);

    // connect(timer, &QTimer::timeout, this, &detectionBoard::_show_src_camera_result);

    // timer->start(1000);

    _video_infer_thread = new video_infer_thread(0, this);

    connect( _video_infer_thread
           , &video_infer_thread::_frame_processed
           , this
           , &detectionBoard::_show_ifd_camera_result
           );

    connect (_video_infer_thread
            , &video_infer_thread::update_border_color_request
            , _video_infer_thread
            , &video_infer_thread::handle_border_color_change
            , Qt::QueuedConnection
            ) ;
    
    connect (_video_infer_thread
            , &video_infer_thread::update_font_type_request
            , _video_infer_thread
            , &video_infer_thread::handle_font_type_change
            , Qt::QueuedConnection
            ) ;

    connect (_video_infer_thread
            , &video_infer_thread::update_font_color_request
            , _video_infer_thread
            , &video_infer_thread::handle_font_color_change
            , Qt::QueuedConnection
            ) ;

    connect (_video_infer_thread
            , &video_infer_thread::update_filling_color_request
            , _video_infer_thread
            , &video_infer_thread::handle_filling_color_change
            , Qt::QueuedConnection
            ) ;
    connect ( _video_infer_thread
            , &video_infer_thread::update_label_position_request
            , _video_infer_thread
            , &video_infer_thread::handle_label_position_change
            ) ;
    
    connect ( _video_infer_thread
            , &video_infer_thread::update_score_threshold_request
            , _video_infer_thread
            , &video_infer_thread::handle_score_threshold_change
            ) ;

    connect ( _video_infer_thread
            , &video_infer_thread::update_nms_1st_request
            , _video_infer_thread
            , &video_infer_thread::handle_nms_1st_threshold_change);

    connect ( _video_infer_thread
            , &video_infer_thread::result_ready
            , this, [this](const detection_result& dr) {
                for (int i = 0; i < dr.class_ids.size(); i++) {
                    const int row = m_det_result_table->rowCount();
                    m_det_result_table->insertRow(row);

                    auto class_item = new QTableWidgetItem(QString(m_labels[dr.class_ids[i]]));
                    auto conf_item = new QTableWidgetItem(QString::number(dr.scores[i], 'f', 2));
                    auto pos_item = new QTableWidgetItem(QString("%1, %2")
                                                                        .arg(dr.boxes[i].x())
                                                                        .arg(dr.boxes[i].y()));
                    auto size_item = new QTableWidgetItem(QString("%1x%2")
                                                                        .arg(dr.boxes[i].width())
                                                                        .arg(dr.boxes[i].height()));

                    m_det_result_table->setItem(row, 0, class_item);
                    m_det_result_table->setItem(row, 1, conf_item);
                    m_det_result_table->setItem(row, 2, pos_item);
                    m_det_result_table->setItem(row, 3, size_item);
                }
            }, Qt::QueuedConnection);

    connect ( _video_infer_thread
            , &video_infer_thread::setup_labels
            , this, [this](const QStringList& labels) {
                m_labels = labels;
            });

    _video_infer_thread->start();

    _push_button_start_video->setEnabled(false);
    _push_button_stop_video->setEnabled(true);

}

void detectionBoard::_on_stop_video_clicked(){
    // _cap.release();
    // _video_timer->stop();

    if (_video_infer_thread) {
        _video_infer_thread->stop();
        _video_infer_thread->quit();
        _video_infer_thread->wait();

        delete _video_infer_thread;
        _video_infer_thread = nullptr;
    }
    _push_button_start_video->setEnabled(true);
    _push_button_stop_video->setEnabled(false);
    // diasable corresponding buttons when stop
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

    QPixmap scaled_pixmap = tmp.scaled( tmp.width() 
                                      , tmp.height() 
                                      , Qt::KeepAspectRatio
                                      , Qt::SmoothTransformation
                                      );

    label->setPixmap(scaled_pixmap);
    label->setScaledContents(false);
}


