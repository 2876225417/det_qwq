#include "dashboard.h"
#include "database.h"
#include <iostream>
#include <qboxlayout.h>
#include <qcombobox.h>
#include <qgroupbox.h>
#include <qlabel.h>
#include <qoverload.h>
#include <qvariant.h>




Dashboard::Dashboard(QWidget *parent)
    : QWidget{parent}
    // dashboard
    , dashboard{new QHBoxLayout()}
    // --camera and config
    , camera_and_config_panel{new QGroupBox("Camera")}
    , camera_and_config_panel_wrapper{new QVBoxLayout()}
    // --camera
    , camera_panel{new QWidget()}
    , camera_wrapper{new QVBoxLayout()}
    , camera_label{new QLabel("Initializing camera...")}
    // --config
    , camera_config_panel{new QGroupBox("Camera Config")}
    , camera_config_layout_wrapper{new QHBoxLayout()}
    // --left
    , camera_left_config_layout{new QVBoxLayout()}
    // ----cuda_enabel_and_camera_select
    , enable_cuda_and_camera_select_wrapper{new QHBoxLayout()}
    // ----cuda enable
    , check_cuda_enable_wrapper{new QHBoxLayout()}
    , check_cuda_enable_label{new QLabel("Enable CUDA: ")}
    , check_cuda_enable_checkbox{new QCheckBox()}
    // ----camera select
    , select_camera_wrapper{new QHBoxLayout()}
    , select_camera_label{new QLabel("Camera")}
    , select_camera_combobox{new QComboBox()}
    // ----contrast select
    , select_contrast_wrapper{new QHBoxLayout()}
    , select_contrast_label{new QLabel("Contrast")}
    , select_contrast_combobox{new QComboBox()}
    // -----gamma select
    , select_gamma_wrapper{new QHBoxLayout()}
    , select_gamma_label{new QLabel("Gamma")}
    , select_gamma_combobox{new QComboBox()}
    // -----brightness select
    , select_brightness_wrapper{new QHBoxLayout()}
    , select_brightness_label{new QLabel("Brightness")}
    , select_brightness_combobox{new QComboBox()}
    // -----saturation select
    , select_saturation_wrapper{new QHBoxLayout()}
    , select_saturation_label{new QLabel("Saturation")}
    , select_saturation_combobox{new QComboBox()}
    // -----denoise params select
    , select_denoise_params_wrapper{new QHBoxLayout()}
    , select_denoise_params_box{new QGroupBox()}
    , select_denoise_params_subwrapper{new QHBoxLayout()}
    // -------denoise type select
    , select_denoise_type_wrapper{new QHBoxLayout()}
    , select_denoise_type_label{new QLabel("Dn-Tp")}
    , select_denoise_type_combobox{new QComboBox()}
    // -------denoise strength select 
    , select_denoise_strength_wrapper{new QHBoxLayout()}
    , select_denoise_strength_label{new QLabel("Dn-Sg")}
    , select_denoise_strength_combobox{new QComboBox()}
    // --right
    , camera_right_config_layout{new QVBoxLayout()}
    // -----exposure params select
    , select_exposure_params_wrapper{new QHBoxLayout()}
    , select_exposure_params_box{new QGroupBox("Exposure")}
    , select_exposure_params_subwrapper{new QHBoxLayout()}
    // -------exposure gamma select
    , select_exposure_gamma_wrapper{new QHBoxLayout()}
    , select_exposure_gamma_label{new QLabel("Exp-Gamma")}
    , select_exposure_gamma_combobox{new QComboBox()}
    // -------exposure gain select
    , select_exposure_gain_wrapper{new QHBoxLayout()}
    , select_exposure_gain_label{new QLabel("Exp-Gain")}
    , select_exposure_gain_combobox{new QComboBox()}
    // ----sharpen params select
    , select_sharpen_params_wrapper{new QHBoxLayout()}
    , select_sharpen_params_box{new QGroupBox("Sharpen")}
    , select_sharpen_params_subwrapper{new QHBoxLayout()}
    // ------sharpen type select
    , select_sharpen_type_wrapper{new QHBoxLayout()}
    , select_sharpen_type_label{new QLabel("Sharpen Type")}
    , select_sharpen_type_combobox{new QComboBox()}
    // ------sharpen strength select
    , select_sharpen_strength_wrapper{new QHBoxLayout()}
    , select_sharpen_strength_label{new QLabel("Sharpen")}
    , select_sharpen_strength_combobox{new QComboBox()}
    // -----super resolution params select
    , select_super_resolution_params_wrapper{new QHBoxLayout()}
    , select_super_resolution_params_box{new QGroupBox("Super Resolution")}
    , select_super_resolution_params_subwrapper{new QHBoxLayout()}
    #ifdef ENABLE__SR_DL__
    // -------super resolution by dl params select
    , select_super_resolution_dl_params_wrapper{new QHBoxLayout()}
    , select_super_resolution_dl_label{new QLabel("SR-DL")}
    , select_super_resolution_dl_combobox{new QComboBox()}
    #endif
    // -------super resolution by gr params select
    , select_super_resolution_gr_params_wrapper{new QHBoxLayout()}
    , select_super_resolution_gr_label{new QLabel("Super Resolution")}
    , select_super_resolution_gr_combobox{new QComboBox()}
    // ------super resolution scale select
    , select_super_resolution_scale_wrapper{new QHBoxLayout()}
    , select_super_resolution_scale_label{new QLabel("Scale")}
    , select_super_resolution_scale_combobox{new QComboBox()}
    {
    QGroupBox* sub_right_sidebar_box = new QGroupBox("Data");   // 数据区 group box

    // 摄像头及配置区域
    // 摄像头区域
    camera_wrapper->addWidget(camera_label, Qt::KeepAspectRatioByExpanding);
    camera_panel->setLayout(camera_wrapper);
    camera_panel->setStyleSheet("border: 1px solid #ccc; border-radius: 6px;");
    camera_panel->setFixedSize(990, 750);
    // 配置区域
    // 左半区
    // --启用CUDA 和 选择摄像头
    // ----启用 CUDA
    m_cuda_enabled = m_settings.value("CUDA/Enabled", false).toBool();
    check_cuda_enable_checkbox->setChecked(m_cuda_enabled);
    check_cuda_enable_wrapper->addWidget(check_cuda_enable_label);
    check_cuda_enable_wrapper->addWidget(check_cuda_enable_checkbox);
    connect(check_cuda_enable_checkbox, &QCheckBox::toggled, this, &Dashboard::on_cuda_toggled);
    // ----选择摄像头
    init_cam_selector();
    select_camera_wrapper->addWidget(select_camera_label);
    select_camera_wrapper->addWidget(select_camera_combobox);
    enable_cuda_and_camera_select_wrapper->addLayout(check_cuda_enable_wrapper);
    enable_cuda_and_camera_select_wrapper->addLayout(select_camera_wrapper);
    

    // --选择对比度
    for (float c = 0.5f; c <= 2.f; c += 0.1f) 
        select_contrast_combobox->addItem(QString::number(c, 'f', 1), c);
    select_contrast_combobox->setCurrentText("1.0");
    connect( select_contrast_combobox
           , QOverload<int>::of(&QComboBox::currentIndexChanged)
           , this
           , &Dashboard::on_contrast_changed
           ) ;
    select_contrast_wrapper->addWidget(select_contrast_label);
    select_contrast_wrapper->addWidget(select_contrast_combobox);
    // --选择gamma值
    for (float g = 0.1f; g <= 3.f; g += 0.3f)
        select_gamma_combobox->addItem(QString::number(g, 'f', 1), g);
    select_gamma_combobox->setCurrentText("1.0");
    connect( select_gamma_combobox
           , QOverload<int>::of(&QComboBox::currentIndexChanged)
           , this
           , &Dashboard::on_gamma_changed
           ) ;
    select_gamma_wrapper->addWidget(select_gamma_label);
    select_gamma_wrapper->addWidget(select_gamma_combobox);
    // --选择亮度值
    for (int b = -100; b <= 100; b += 20) 
        select_brightness_combobox->addItem(QString("%1%").arg(b), static_cast<int>(2.55f * b));
    select_brightness_combobox->setCurrentText("0%");
    connect( select_brightness_combobox
           , QOverload<int>::of(&QComboBox::currentIndexChanged)
           , this
           , &Dashboard::on_brightness_changed
           ) ;
    select_brightness_wrapper->addWidget(select_brightness_label);
    select_brightness_wrapper->addWidget(select_brightness_combobox);
    // --选择饱和度
    for (float s = 0.1f; s <= 2.f; s += 0.3f)
        select_saturation_combobox->addItem(QString::number(s, 'f', 1), s);
    select_saturation_combobox->setCurrentText("1.0");
    connect( select_saturation_combobox
           , QOverload<int>::of(&QComboBox::currentIndexChanged)
           , this
           , &Dashboard::on_saturation_changed
           ) ;
    select_saturation_wrapper->addWidget(select_saturation_label);
    select_saturation_wrapper->addWidget(select_saturation_combobox);
    // --降噪参数
    select_denoise_type_combobox->addItem("OFF", QVariant::fromValue(denoise_type::NONE));
    select_denoise_type_combobox->addItem("Gaussian", QVariant::fromValue(denoise_type::GAUSSIAN));
    select_denoise_type_combobox->addItem("Bilateral", QVariant::fromValue(denoise_type::BILATERAL));
    select_denoise_type_combobox->addItem("NLMEANS", QVariant::fromValue(denoise_type::NLMEANS));
    select_denoise_type_combobox->addItem("MEDIAN", QVariant::fromValue(denoise_type::MEDIAN));
    select_denoise_type_combobox->setCurrentText("OFF");
    connect(select_denoise_type_combobox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &Dashboard::on_denoise_type_changed);
    
    for (float d = 0.f; d <= 3.f; d += 0.2f)
        select_denoise_strength_combobox->addItem(QString::number(d, 'f', 1), d);
    select_denoise_strength_combobox->setCurrentText("0.0");
    connect(select_denoise_strength_combobox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &Dashboard::on_denoise_strength_changed);
    select_denoise_type_wrapper->addWidget(select_denoise_type_label);
    select_denoise_type_wrapper->addWidget(select_denoise_type_combobox);
    select_denoise_strength_wrapper->addWidget(select_denoise_strength_label);
    select_denoise_strength_wrapper->addWidget(select_denoise_strength_combobox);

    select_denoise_params_subwrapper->addLayout(select_denoise_type_wrapper);
    select_denoise_params_subwrapper->addLayout(select_denoise_strength_wrapper);
    select_denoise_params_box->setLayout(select_denoise_params_subwrapper);
    select_denoise_params_wrapper->addWidget(select_denoise_params_box);

    camera_left_config_layout->addLayout(enable_cuda_and_camera_select_wrapper);// CUDA and camera
    camera_left_config_layout->addLayout(select_contrast_wrapper);              // contrast
    camera_left_config_layout->addLayout(select_gamma_wrapper);                 // gamma
    camera_left_config_layout->addLayout(select_brightness_wrapper);            // brightness
    camera_left_config_layout->addLayout(select_saturation_wrapper);            // saturation
    camera_left_config_layout->addLayout(select_denoise_params_wrapper);        // denoise

    // 右半区
    // --选择曝光强度
    for (float e_gamma = 0.2f; e_gamma <= 2.f; e_gamma += 0.2f)
        select_exposure_gamma_combobox->addItem(QString::number(e_gamma, 'f', 1), e_gamma);
    select_exposure_gamma_combobox->setCurrentText("1.0");
    connect(select_exposure_gamma_combobox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &Dashboard::on_exp_gamma_changed);

    for (float e_gain = 0.2f; e_gain <= 2.f; e_gain += 0.2f)
        select_exposure_gain_combobox->addItem(QString::number(e_gain, 'f', 1), e_gain);
    select_exposure_gain_combobox->setCurrentText("1.0");
    connect(select_exposure_gain_combobox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &Dashboard::on_exp_gain_changed);

    select_exposure_gamma_wrapper->addWidget(select_exposure_gamma_label);
    select_exposure_gamma_wrapper->addWidget(select_exposure_gamma_combobox);
    select_exposure_gain_wrapper->addWidget(select_exposure_gain_label);
    select_exposure_gain_wrapper->addWidget(select_exposure_gain_combobox);
    select_exposure_params_subwrapper->addLayout(select_exposure_gamma_wrapper);
    select_exposure_params_subwrapper->addLayout(select_exposure_gain_wrapper);
    select_exposure_params_box->setLayout(select_exposure_params_subwrapper);
    select_exposure_params_wrapper->addWidget(select_exposure_params_box);
    // --选择锐化类型
    select_sharpen_type_combobox->addItem("Laplacian", QVariant::fromValue(sharpen_type::LAPLACIAN));
    select_sharpen_type_combobox->addItem("Unsharp Mask", QVariant::fromValue(sharpen_type::UNSHARP_MASK));
    select_sharpen_type_combobox->addItem("Edge Enhance", QVariant::fromValue(sharpen_type::EDGE_ENHANCE));
    select_sharpen_type_combobox->setCurrentText("Laplacian");
    connect(select_sharpen_type_combobox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &Dashboard::on_sharpen_type_changed);
    select_sharpen_type_wrapper->addWidget(select_sharpen_type_label);
    select_sharpen_type_wrapper->addWidget(select_sharpen_type_combobox);
    // --选择锐化程度
    for (float s = 0.f; s <= 3.f; s += 0.5f)
        select_sharpen_strength_combobox->addItem(QString::number(s, 'f', 1), s);
    select_sharpen_strength_combobox->setCurrentText("0.0");
    connect( select_sharpen_strength_combobox
           , QOverload<int>::of(&QComboBox::currentIndexChanged)
           , this
           , &Dashboard::on_sharpen_strength_changed
           ) ;
    select_sharpen_strength_wrapper->addWidget(select_sharpen_strength_label);
    select_sharpen_strength_wrapper->addWidget(select_sharpen_strength_combobox);

    select_sharpen_params_subwrapper->addLayout(select_sharpen_type_wrapper);
    select_sharpen_params_subwrapper->addLayout(select_sharpen_strength_wrapper);
    select_sharpen_params_box->setLayout(select_sharpen_params_subwrapper);
    select_sharpen_params_wrapper->addWidget(select_sharpen_params_box);

    // --超分类型
    // ----一般方法
    select_super_resolution_gr_combobox->addItem("OFF", QVariant::fromValue(sr_type::NONE));
    select_super_resolution_gr_combobox->addItem("BILINEAR", QVariant::fromValue(sr_type::BILINEAR));
    select_super_resolution_gr_combobox->addItem("BICUBIC", QVariant::fromValue(sr_type::BICUBIC));
    select_super_resolution_gr_combobox->addItem("LANCZOS", QVariant::fromValue(sr_type::LANCZOS));
    select_super_resolution_gr_combobox->addItem("EDGE_AWARE", QVariant::fromValue(sr_type::EDGE_AWARE));
    select_super_resolution_gr_combobox->addItem("WNNM", QVariant::fromValue(sr_type::WNNM));
    select_super_resolution_gr_combobox->setCurrentText("OFF");
    connect(select_super_resolution_gr_combobox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &Dashboard::on_sr_gr_changed);
    // ----放大倍率
    select_super_resolution_scale_combobox->addItem("2", QVariant::fromValue(2));
    select_super_resolution_scale_combobox->addItem("4", QVariant::fromValue(4));
    select_super_resolution_scale_combobox->setCurrentText("2");
    connect(select_super_resolution_scale_combobox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &Dashboard::on_sr_scale_changed);

    // ----深度学习
    #ifdef ENABLE__SR_DL__
    select_super_resolution_dl_combobox->addItem("OFF", QVariant::fromValue(sr_dl_type::NONE));
    select_super_resolution_dl_combobox->addItem("ESDR", QVariant::fromValue(sr_dl_type::EDSR));
    select_super_resolution_dl_combobox->addItem("ESPCN", QVariant::fromValue(sr_dl_type::ESPCN));
    select_super_resolution_dl_combobox->addItem("FSRCNN", QVariant::fromValue(sr_dl_type::FSRCNN));
    select_super_resolution_dl_combobox->addItem("LAPSRN", QVariant::fromValue(sr_dl_type::LAPSRN));
    select_super_resolution_dl_combobox->setCurrentText("OFF");
    connect(select_super_resolution_dl_combobox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &Dashboard::on_sr_dl_changed);
    #endif
    
    select_super_resolution_scale_wrapper->addWidget(select_super_resolution_scale_label);
    select_super_resolution_scale_wrapper->addWidget(select_super_resolution_scale_combobox);
    select_super_resolution_gr_params_wrapper->addWidget(select_super_resolution_gr_label);
    select_super_resolution_gr_params_wrapper->addWidget(select_super_resolution_gr_combobox);
    #ifdef ENABLE__SR_DL__
    select_super_resolution_dl_params_wrapper->addWidget(select_super_resolution_dl_label);
    select_super_resolution_dl_params_wrapper->addWidget(select_super_resolution_dl_combobox);
    select_super_resolution_params_subwrapper->addLayout(select_super_resolution_dl_params_wrapper);
    #endif
    select_super_resolution_params_subwrapper->addLayout(select_super_resolution_gr_params_wrapper);
    select_super_resolution_params_subwrapper->addLayout(select_super_resolution_scale_wrapper);
    select_super_resolution_params_box->setLayout(select_super_resolution_params_subwrapper);
    select_super_resolution_params_wrapper->addWidget(select_super_resolution_params_box);

    camera_right_config_layout->addLayout(select_exposure_params_wrapper);          // exposure
    camera_right_config_layout->addLayout(select_sharpen_params_wrapper);           // sharpen
    camera_right_config_layout->addLayout(select_super_resolution_params_wrapper);  // super resolution

    camera_config_layout_wrapper->addLayout(camera_left_config_layout);             // left config
    camera_config_layout_wrapper->addLayout(camera_right_config_layout);            // right config
    camera_config_panel->setLayout(camera_config_layout_wrapper);
    QVBoxLayout* camera_and_config_panel_wrapper = new QVBoxLayout();
    camera_and_config_panel_wrapper->addWidget(camera_panel);
    camera_and_config_panel_wrapper->addWidget(camera_config_panel);
    camera_and_config_panel->setLayout(camera_and_config_panel_wrapper);

    // 用户编辑区
    // 1. 图片
    ItemImage = new QLabel(this);
    ItemImage->setFixedSize(150, 150);
    ItemImage->setStyleSheet("background-color: gray");
    // 2. 数据
    ItemName = new QLineEdit(this);
    connect(ItemName, &QLineEdit::textChanged, this, &Dashboard::onItemNameChanged);
    ItemLength = new QLineEdit(this);
    ItemWidth = new QLineEdit(this);
    ItemHeight = new QLineEdit(this);
    classIdLabel = new QLabel("None", this);
    classIdLabel->setFixedWidth(80);
    // 3. 保存按钮
    save = new QPushButton("save", this);
    save->setFixedWidth(50);
    connect(save, &QPushButton::clicked, this, &Dashboard::saveItemInfo);
    // 添加到表单组件中
    QFormLayout* item_data_form = new QFormLayout();
    item_data_form->addRow("Item Name: ", ItemName);
    item_data_form->addRow("Length(cm): ", ItemLength);
    item_data_form->addRow("Width(cm): ", ItemWidth);
    item_data_form->addRow("Height(cm): ", ItemHeight);
    item_data_form->addRow("Class ID: ", classIdLabel);

    QGroupBox* data_edit = new QGroupBox("DataEdit");
    QHBoxLayout* userAreaLayout = new QHBoxLayout();
    userAreaLayout->addWidget(ItemImage, 1);
    userAreaLayout->addLayout(item_data_form, 2);
    userAreaLayout->addWidget(save, 1);

    data_edit->setLayout(userAreaLayout);
    data_edit->setStyleSheet("padding: 3px");

    // 已经收集信息区域
    scrollArea = new QScrollArea(this);
    scrollWidget = new QWidget(scrollArea);
    galleryLayout = new QGridLayout(scrollWidget);
    scrollWidget->setLayout(galleryLayout);
    scrollArea->setWidget(scrollWidget);
    scrollArea->setWidgetResizable(true);

    QHBoxLayout* collected_data = new QHBoxLayout();
    QGroupBox* collected_data_box = new QGroupBox("CollectedData");
    collected_data->addWidget(scrollArea);
    collected_data_box->setLayout(collected_data);


    QVBoxLayout* collected_data_wrapper = new QVBoxLayout();
    collected_data_wrapper->setSpacing(10);                     // 设置组件间的间隔
    collected_data_wrapper->setContentsMargins(5, 5, 5, 5);     // 设置内边距
    collected_data_wrapper->addWidget(data_edit);               // 数据编辑区
    collected_data_wrapper->addWidget(collected_data_box);      // 采集的数据展示区
    sub_right_sidebar_box->setLayout(collected_data_wrapper);

    // 整个 dashboard 左边(3/4)为摄像头区域 右边(1/4)为采集的数据部分
    dashboard->addWidget(camera_and_config_panel, 3);
    dashboard->addWidget(sub_right_sidebar_box, 1);
    setLayout(dashboard);

    connect(select_camera_combobox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &Dashboard::on_camera_change);
}

Dashboard::~Dashboard(){

}

// 左半区配置
void Dashboard::on_cuda_toggled(bool checked) {
    m_cuda_enabled = checked;
    m_settings.setValue("CUDA/Enabled", checked);

    if(m_cam) 
        m_cam->set_cuda_enabled(checked);
}

void Dashboard::on_camera_change(int idx) {
    QVariant selected_idx = select_camera_combobox->itemData(idx);

    if (!selected_idx.isValid()) {
        qWarning() << "Invalid selection of camera index!";
        return;
    }

    int opt_type = selected_idx.toInt();

    if (opt_type == -1) {
        stop_current_cam();
        camera_label->setPixmap(QPixmap());
        camera_label->setText("<font color='gray'>Camera is off.</font>");
        return;
    }

    int cam_idx = selected_idx.toInt();
    stop_current_cam();

    try {
        m_cam = std::make_unique<camera_capturer>(cam_idx, m_cuda_enabled);
        connect( m_cam.get()
               , &camera_capturer::frame_captured
               , this
               , &Dashboard::updateFrame
               ) ;

        connect( m_cam.get()
               , &camera_capturer::error_occurred
               , this
               , &Dashboard::on_camera_error_occur
               ) ;
        m_cam->start();

    } catch (const std::exception& e) {
        QMessageBox::critical( this
                             , "error"
                             , QString("Failed to initialize camera: %1").arg(e.what())
                             ) ;
    }
}

void Dashboard::on_camera_error_occur(const QString& msg) {
    QMessageBox::warning(this, "Camera error!", msg);
    stop_current_cam();
    camera_label->setText("Camera connection failed!"); 
}

void Dashboard::on_contrast_changed(int index) {
    float contrast = select_contrast_combobox->itemData(index).toFloat();
    m_contrast = contrast;
    
    if (m_cam) 
        m_cam->set_contrast(contrast);
    qDebug() << "contrast: " << contrast;
}

void Dashboard::on_gamma_changed(int index) {
    float gamma = select_gamma_combobox->itemData(index).toFloat();
    
    if (m_cam) 
        m_cam->set_gamma(gamma);
}

void Dashboard::on_brightness_changed(int index) { 
    int brightness = select_brightness_combobox->itemData(index).toInt();
    qDebug() << "Brightness: " << brightness;
    if(m_cam)
        m_cam->set_brightness(brightness);
}

void Dashboard::on_saturation_changed(int index) {
    float saturation = select_saturation_combobox->itemData(index).toFloat();

    if (m_cam) 
        m_cam->set_saturation(saturation);
}

void Dashboard::on_denoise_type_changed(int index) {
    denoise_type type = select_denoise_type_combobox->itemData(index).value<denoise_type>();

    if (m_cam)
        m_cam->set_denoise_type(type);

}

void Dashboard::on_denoise_strength_changed(int index) {
    float denoise_strength = select_denoise_strength_combobox->itemData(index).toFloat();

    if (m_cam)
        m_cam->set_denoise_strength(denoise_strength);
}


// 右半区配置
void Dashboard::on_exp_gamma_changed(int index) {
    float exp_gamma = select_exposure_gamma_combobox->itemData(index).toFloat();
    
    if (m_cam)
        m_cam->set_exp_gamma(exp_gamma);
}

void Dashboard::on_exp_gain_changed(int index) {
    float exp_gain = select_exposure_gamma_combobox->itemData(index).toFloat();
    
    if (m_cam)
        m_cam->set_exp_gain(exp_gain);
}

void Dashboard::on_sharpen_type_changed(int index) {
    sharpen_type type = select_sharpen_type_combobox->itemData(index).value<sharpen_type>();

    if (m_cam) 
        m_cam->set_sharpen_type(type);
}

void Dashboard::on_sharpen_strength_changed(int index) {
    float sharpen_strength = select_sharpen_strength_combobox->itemData(index).toFloat();
    
    if (m_cam)
        m_cam->set_sharpen_strength(sharpen_strength);
    qDebug() << "sharpen strength: " << sharpen_strength;
}

void Dashboard::on_sr_gr_changed(int index) {
    sr_type super_resolution_gr_type = select_super_resolution_gr_combobox->itemData(index).value<sr_type>();

    if (m_cam) 
        m_cam->set_super_resolution_gr_type(super_resolution_gr_type);
}

#ifdef ENABLE__SR_DL__
void Dashboard::on_sr_dl_changed(int index) {
    sr_dl_type super_resolution_dl_type = select_super_resolution_dl_combobox->itemData(index).value<sr_dl_type>();

    if (m_cam)
        m_cam->set_super_resolution_dl_type(super_resolution_dl_type);
}
#endif

void Dashboard::on_sr_scale_changed(int index) {
    int scale = select_super_resolution_scale_combobox->itemData(index).toInt();

    qDebug() << "scale: " << scale;
    if (m_cam)
        m_cam->set_super_resolution_gr_scale(scale);
}

void
Dashboard::onItemNameChanged(const QString& text){
    if(text.isEmpty()){
        classIdLabel->setText("None");
        return;
    }

    auto [classId, amount] = dbConn::instance().onItemNameChangedDb(text);

    if(classId != -1){
        classIdLabel->setText(QString::number(classId));
    } else {
        classIdLabel->setText("No such class");
    }
}

void 
Dashboard::updateFrame(const QImage& img) {

    if (!m_fps_timer.isValid())
        m_fps_timer.start();

    m_frame_counter++;

    if (m_fps_timer.elapsed() > 1000) {
        m_current_fps = m_frame_counter / (m_fps_timer.elapsed() / 1000.f);
        m_frame_counter = 0;
        m_fps_timer.restart();
    }

    current_cap_frame = img;
    QPixmap pixmap = QPixmap::fromImage(img);
    QPainter painter(&pixmap);
    draw_fps_overlay(painter);
    painter.setPen(QPen(Qt::white, 2, Qt::DashLine));
    QRect frameRect = mapRectToFrame(currentSelection);
    painter.drawRect(frameRect);

    if(isSelecting){
        QRect tempRect = QRect(selectionStart, selectionEnd).normalized();
        QRect tempFrameRect = mapRectToFrame(tempRect);
        painter.drawRect(tempFrameRect);
    }

    camera_label->setPixmap(pixmap.scaled(camera_label->size(), Qt::KeepAspectRatio));
}

void Dashboard::draw_fps_overlay(QPainter& painter) {
    painter.setRenderHint(QPainter::Antialiasing);

    painter.setBrush(QColor(0, 0, 0, 150));
    painter.drawRoundedRect(10, 10, 120, 40, 5, 5);

    // dynamic font color by fps
    QColor text_color;
    if (m_current_fps > 25) text_color = Qt::green;
    else if (m_current_fps > 15) text_color = Qt::yellow;
    else text_color = Qt::red;

    static qreal angle = 0;
    QLinearGradient grad(0, 0, 100, 0);
    grad.setColorAt(0, text_color);
    grad.setColorAt(1, text_color.darker(150));
    grad.setSpread(QGradient::RepeatSpread);

    QFont font("Monospace", 10, QFont::Bold);
    painter.setFont(font);
    painter.setPen(QPen(grad, 2));

    const QString fps_text = QString("FPS: %1").arg(m_current_fps, 5, 'f', 1);
    painter.drawText(20, 30, fps_text);
}








///////////////////////////////////////////
//                                       //
//                                       //
//        Camera Config                  //
//                                       //
//                                       //
///////////////////////////////////////////
void Dashboard::init_cam_selector() {
    select_camera_combobox->clear();
    // find_available_cameras
    auto cameras = []() -> QVector<int> {
        QVector<int> availables;
        for (int i = 0; i < 10; ++i) {
            cv::VideoCapture probe(i);
            if (probe.isOpened()) {
                availables.append(i);
                probe.release();
                QThread::msleep(100);
            }
        }
        return availables;
    }();

    for (int idx: cameras) 
        select_camera_combobox->addItem(QString("Camera %1").arg(idx), idx);
    
    select_camera_combobox->addItem("Off", QVariant(-1));
}


void 
Dashboard::stop_current_cam() {
    if (m_cam) {
        m_cam->stop();
        m_cam->disconnect();
        m_cam.reset();
    }
}



QVector<int> Dashboard::find_available_cams() const {
    QVector<int> availables;
    for (int i = 0; i < 10; ++i) {
        cv::VideoCapture probe(i);
        if (probe.isOpened()) {
            availables.append(i);
            probe.release();
            QThread::msleep(100);
        }
    }
    return availables;
}




// 点击 save 后保存至 gallery 中
void
Dashboard::addRegionToGallery( const QPixmap& pixmap
                             , const QString& id
                             , const QString& name
                             , const QString& length
                             , const QString& width
                             , const QString& height
                             ) {

    ItemInfo* item = new ItemInfo( pixmap
                                 , id
                                 , name
                                 , length
                                 , width
                                 , height
                                 , this
                                 );
    Items.append(item);

    galleryLayout->addWidget(item);
    // updateUserArea(item);
}


void
Dashboard::mouseReleaseEvent(QMouseEvent* event){
    if(event->button() == Qt::LeftButton){
        if(isSelecting){
            isSelecting = false;

            selectionEnd = event->pos() - camera_label->geometry().topLeft();
            QRect rect = QRect(selectionStart, selectionEnd).normalized();
                  rect = rect.intersected(QRect(QPoint(0, 0), camera_label->size()));

            if(rect.width() > 0 && rect.height() > 0){
                QRect frameRect = mapRectToFrame(rect);                
                if (!current_cap_frame.isNull()) {
                    QPixmap pixmap = QPixmap::fromImage(current_cap_frame);

                    if (frameRect.x() >= 0 &&
                        frameRect.y() >= 0 &&
                        frameRect.x() + frameRect.width() <= pixmap.width() &&
                        frameRect.y() + frameRect.height() <= pixmap.height()) {
                        QPixmap cropped_pixmap = pixmap.copy(frameRect);
                        QPixmap scaled_pixmap = cropped_pixmap.scaled( 150
                                                                     , 150
                                                                     , Qt::KeepAspectRatioByExpanding
                                                                     , Qt::SmoothTransformation
                                                                     ) ;

                        ItemImage->setPixmap(scaled_pixmap);
                        currentSelection = rect;
                    } else {

                    }
                } else {

                }
            }
        } else if(isDragging){
            isDragging = false;
        }
    }
}

// 鼠标点位相对图像位置映射
QPoint
Dashboard::mapToFrame(const QPoint& widgetPoint){
    QSize video_size = camera_label->pixmap().size();
    QSize label_size = camera_label->size();
    QRect display_rect = camera_label->contentsRect();

    double xScale = double(video_size.width()) / display_rect.width();
    double yScale = double(video_size.height()) / display_rect.height();

    int offsetX = (label_size.width() - display_rect.width()) / 2;
    int offsetY = (label_size.height() - display_rect.height()) / 2;

    int frameX = (widgetPoint.x() - offsetX) * xScale;
    int frameY = (widgetPoint.y() - offsetY) * yScale;

    return QPoint(frameX, frameY);
}

QRect
Dashboard::mapRectToFrame(const QRect& widgetRect){
    QPoint topLeft = mapToFrame(widgetRect.topLeft());
    QPoint bottomRight = mapToFrame(widgetRect.bottomRight());
    return QRect(topLeft, bottomRight);
}



bool
Dashboard::saveItemInfo(){
    QString name = ItemName->text();
    QString length = ItemLength->text();
    QString width = ItemWidth->text();
    QString height = ItemHeight->text();
    QPixmap image = ItemImage->pixmap().copy();

    // QPixmap imageStore = ItemImage->pixmap();
    if(image.isNull()){
        QMessageBox::warning(this, "Error", "Please select a valid image");
        return false;
    }

    auto [classId, amount] = dbConn::instance().onItemNameChangedDb(name);

    qDebug() << "classId: " << classId;

    if(classId == -1){
        classId = dbConn::instance().insertNewClassId(name);
        amount = 0;
    } else {
        if(!dbConn::instance().updateClassAmount(classId)){
            return false;
        }
    }

    QString uniqueName = name + "_" + QString::number(amount + 1);

    QString saveDir = "dataset/images";
    QDir dir(saveDir);
    if(!dir.exists()){
        dir.mkpath(".");
    }

    QString imagePath = saveDir + "/" + uniqueName + ".jpg";
    image = image.scaled(640, 640);
    image.save(imagePath);

    QString labelDir = "dataset/labels";
    QDir labelDirectory(labelDir);
    if(!labelDirectory.exists()){
        labelDirectory.mkpath(".");
    }

    QString labelPath = labelDir + "/" + uniqueName + ".txt";
    QFile labelFile(labelPath);
    if(labelFile.open(QIODevice::WriteOnly | QIODevice::Text)){
        QTextStream out(&labelFile);

        int classId = dbConn::instance().get_inserted_item_class_id(name);

        float x_center = 0.5;
        float y_center = 0.5;
        float boxWidth = 0.8;
        float boxHeight = 0.8;

        out << classId << " "
            << x_center << " "
            << y_center << " "
            << boxWidth << " "
            << boxHeight << "\n";

        labelFile.close();
    } else {
        QMessageBox::warning(this, "Error", "Failed to create label file");
        return false;
    }

    QFile file(imagePath);
    int size = 0;
    if(file.exists()){
        file.open(QIODevice::ReadOnly);
        size = file.size();
    }

    QString tableName = "dataset";

    // 将数据插入到数据库的表中
    int inserted_id = dbConn::instance().insertUserEdit(tableName, // 表名
                                        name,                   // 名称
                                        length,                 // 长度
                                        width,                  // 宽度
                                        height,                 // 高度
                                        imagePath,              // 图片路径
                                        size);                  // 图片大小

    if(inserted_id != -1){
        QMessageBox::information(this, "Success", "Item saved sucessfully");

        addRegionToGallery(image,
                           QString::number(inserted_id),
                           name,
                           length,
                           width,
                           height);

    } else {
        QMessageBox::warning(this, "Error", "Failed to save item");
    }


    qDebug() << "Item info: "
           << name << ", "
           << length << ", "
           << width << ", "
           << height;

    ItemImage->clear();
    ItemName->clear();
    ItemLength->clear();
    ItemWidth->clear();
    ItemHeight->clear();
    currentSelection = QRect();

    return true;
}


void
Dashboard::mouseMoveEvent(QMouseEvent* event){
    if(isSelecting){
        if(camera_label->geometry().contains(event->pos())){
            selectionEnd = event->pos() - camera_label->geometry().topLeft();
            update();
            qDebug() << "Selected rect: " << currentSelection << '\n';

        }
    } else if(isDragging){
        if(camera_label->geometry().contains(event->pos())){
            QPoint newTopLeft = event->pos() - camera_label->geometry().topLeft() - dragOffset;
            currentSelection.moveTopLeft(newTopLeft);
            update();

            updateItemImage();
        }
    } else {
        if(currentSelection.contains(event->pos()- camera_label->geometry().topLeft())){
            if(!isHoveringOverSelection){
                isHoveringOverSelection = true;
            }
        } else {
            if(isHoveringOverSelection){
                isHoveringOverSelection = false;
            }
        }

    }
}


QImage
Dashboard::cvMatToImage(const cv::Mat& mat){
    return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
}

void
Dashboard::mousePressEvent(QMouseEvent* event){
    if(event->button() == Qt::LeftButton){
        if(currentSelection.contains(event->pos() - camera_label->geometry().topLeft())){
            isDragging = true;
            dragOffset = event->pos() - camera_label->geometry().topLeft() - currentSelection.topLeft();
        }
        else if(camera_label->geometry().contains(event->pos())){
            currentSelection = QRect();
            isSelecting = true;
            selectionStart = event->pos() - camera_label->geometry().topLeft();
            selectionEnd = selectionStart;
        }
    }
}

void Dashboard::updateItemImage() {
    if (currentSelection.isValid()) {
        QRect frameRect = mapRectToFrame(currentSelection);

        if (!current_cap_frame.isNull()) {
            QPixmap pixmap = QPixmap::fromImage(current_cap_frame);

            if (frameRect.x() >= 0 &&
                frameRect.y() >= 0 &&
                frameRect.x() + frameRect.width() <= pixmap.width() &&
                frameRect.y() + frameRect.height() <= pixmap.height()) {
                    QPixmap cropped_pixmap = pixmap.copy(frameRect);

                    QPixmap scaled_pixmap = cropped_pixmap.scaled( 150
                                                                 , 150
                                                                 , Qt::KeepAspectRatioByExpanding
                                                                 , Qt::SmoothTransformation
                                                                 ) ;

                    ItemImage->setPixmap(scaled_pixmap);
                } else {

                }
        } else {

        }
    }
}