#include "train_configuration_panel.h"

#include <csignal>
#include <exception>
#include <memory>
#include <ostream>
#include <qboxlayout.h>
#include <qcheckbox.h>
#include <qcombobox.h>
#include <qdatetime.h>
#include <qdebug.h>
#include <qfiledialog.h>
#include <qfileinfo.h>
#include <qlabel.h>
#include <qlineedit.h>
#include <qnamespace.h>
#include <qoverload.h>
#include <qpushbutton.h>
#include <qspinbox.h>
#include <qstandardpaths.h>
#include <qvariant.h>
#include <stdexcept>
#include <thread>
#include <vector>
#include <algorithm>

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

QComboBox* workers_combobox_with_concurrency(QWidget* parent = nullptr) {
    QComboBox* workers_combobox = new QComboBox(parent);
    typedef unsigned int UI;
    UI hardware_threads = std::thread::hardware_concurrency();

#ifdef DEFAULT_HARDWARE_THREADS
    if (hardware_threads == 0)
        hardware_threads = DEFAULT_HARDWARE_THREADS
#else
    if (hardware_threads == 0)
        hardware_threads = 8;
#endif
            std::vector<UI> thread_opts;
    for (UI i = 1; i <= hardware_threads; i *= 2)
        thread_opts.push_back(i);


    std::for_each( thread_opts.begin()
                 , thread_opts.end()
                 , [workers_combobox](int opt) {
                    workers_combobox->addItem(QString::number(opt)); }
                 ) ;

    auto idx = std::lower_bound( thread_opts.begin()
                               , thread_opts.end()
                               , hardware_threads >> 1
                               ) ;

    int default_idx = (idx != thread_opts.end()) ?
                       std::distance(thread_opts.begin(), idx) :
                       thread_opts.size() - 1;

    workers_combobox->setCurrentIndex(default_idx);
    return workers_combobox;
}


Train_Configuration_Panel::Train_Configuration_Panel(QWidget* parent)
    : QWidget(parent)
    , train_cfg_opts_default{std::make_unique<train_config_options>()}
    , train_cfg_opts{std::make_unique<train_config_options>()}
    , config_panel{new QVBoxLayout()}
    // wrapper
    , config_panel_wrapper{new QVBoxLayout()}
    // row_1
    , basic_and_control_wrapper{new QHBoxLayout()}
    , basic_cfg_box{new QGroupBox("BasicConig")}
    , basic_cfg_layout_wrapper{new QVBoxLayout()}
    , control_cfg_box{new QGroupBox("ControlConfig")}
    , control_cfg_layout_wrapper{new QVBoxLayout()}
    // row_2
    , col_2_wrapper{new QHBoxLayout()}
    , advanced_cfg_box{new QGroupBox("AdvancedConfig")}
    , advanced_col_wrapper{new QHBoxLayout()}
    , advanced_cfg_layout_wrapper_col_1{new QVBoxLayout()}
    // row_3
    , optimizer_and_device_and_strategy_wrapper_col_2{new QVBoxLayout()}
    , optimizer_wrapper{new QVBoxLayout()}
    , optimizer_cfg_box{new QGroupBox("OptimizerConfig")}
    , optimizer_cfg_layout_wrapper{new QVBoxLayout()}
    , device_and_strategy_wrapper{new QVBoxLayout()}
    , device_cfg_box{new QGroupBox("DeviceConfig")}
    , device_cfg_layout_wrapper{new QVBoxLayout()}
    , strategy_cfg_box{new QGroupBox("StrategyConfig")}
    , strategy_cfg_layout_wrapper{new QVBoxLayout()}
    // row_4 
    , row_3_wrapper{new QHBoxLayout()}
    , augmentation_cfg_box{new QGroupBox("AugmentationConfig")}
    , augmentation_cfg_layout_wrapper{new QHBoxLayout()}
    , augmentation_cfg_layout_col_1_wrapper{new QVBoxLayout()}
    , augmentation_cfg_layout_col_2_wrapper{new QVBoxLayout()}
    // row_5
    , extra_and_loss_wrapper{new QVBoxLayout()}
    , loss_cfg_layout_wrapper{new QVBoxLayout()}
    , loss_cfg_box{new QGroupBox("LossConig")}
    , extra_cfg_layout_wrapper{new QVBoxLayout()}
    , extra_cfg_box{new QGroupBox("ExtraConfig")},
    // 
    export_and_reset_button_layout_wrapper{new QHBoxLayout()}
    {
    qDebug() << "\033[31mThis is red text\033[0m";

    // row_1
    // basic
    // --data
    select_data_file_wrapper = new QHBoxLayout();
    select_data_file_button = new QPushButton("DataFile");
    selected_data_file_label = new QLabel("Yaml_Data");
    select_data_file_wrapper->addWidget(selected_data_file_label);
    select_data_file_wrapper->addWidget(select_data_file_button);

    // --cache
    select_cache_type_wrapper = new QHBoxLayout();
    selected_cache_type_label = new QLabel("Cache");
    cache_type_combobox = new QComboBox();
    cache_type_combobox->addItem("None", QVariant::fromValue(Cache_Type::None));
    cache_type_combobox->addItem("RAM",  QVariant::fromValue(Cache_Type::RAM));
    cache_type_combobox->addItem("Disk", QVariant::fromValue(Cache_Type::Disk));
    cache_type_combobox->setCurrentIndex(1);
    select_cache_type_wrapper->addWidget(selected_cache_type_label);
    select_cache_type_wrapper->addWidget(cache_type_combobox);

    // --workers
    select_workers_num_wrapper = new QHBoxLayout();
    selected_workers_num_label = new QLabel("Workers");
    workers_num_combobox = workers_combobox_with_concurrency();
    workers_num_combobox->setCurrentIndex(4);
    select_workers_num_wrapper->addWidget(selected_workers_num_label);
    select_workers_num_wrapper->addWidget(workers_num_combobox);

    // --project
    set_project_alias_wrapper = new QHBoxLayout();
    project_alias_label = new QLabel("Project: ");
    setted_project_alias = new QLineEdit();
    setted_project_alias->setText("runs/train");
    setted_project_alias->setFixedWidth(90);
    set_project_alias_wrapper->addWidget(project_alias_label);
    set_project_alias_wrapper->addWidget(setted_project_alias);

    // --name
    set_name_alias_wrapper = new QHBoxLayout();
    name_alias_label = new QLabel("Name: ");
    setted_name_alias = new QLineEdit();
    setted_name_alias->setText("exp");
    setted_name_alias->setFixedWidth(90);
    set_name_alias_wrapper->addWidget(name_alias_label);
    set_name_alias_wrapper->addWidget(setted_name_alias);

    // --exist_ok
    check_exist_ok_wrapper = new QHBoxLayout();
    exist_ok_label = new QLabel("Exist_Ok: ");
    exist_ok_checkbox = new QCheckBox("overwrite");
    exist_ok_checkbox->setChecked(train_cfg_opts->basic_cfg.exist_ok);
    check_exist_ok_wrapper->addWidget(exist_ok_label);
    check_exist_ok_wrapper->addWidget(exist_ok_checkbox);

    // --fraction
    adjust_fraction_wrapper = new QHBoxLayout();
    fraction_label = new QLabel("Fraction: ");
    fraction_adjuster = new QSlider(Qt::Horizontal);
    fraction_adjuster->setRange(0, 100);
    fraction_adjuster->setValue(100);
    adjust_fraction_wrapper->addWidget(fraction_label);
    adjust_fraction_wrapper->addWidget(fraction_adjuster);

    basic_cfg_layout_wrapper->addLayout(select_data_file_wrapper);          // data
    basic_cfg_layout_wrapper->addLayout(select_cache_type_wrapper);         // cache
    basic_cfg_layout_wrapper->addLayout(select_workers_num_wrapper);        // workers
    basic_cfg_layout_wrapper->addLayout(set_project_alias_wrapper);         // project
    basic_cfg_layout_wrapper->addLayout(set_name_alias_wrapper);            // name
    basic_cfg_layout_wrapper->addLayout(check_exist_ok_wrapper);            // exist_ok
    basic_cfg_layout_wrapper->addLayout(adjust_fraction_wrapper);           // fraction

    basic_cfg_box->setLayout(basic_cfg_layout_wrapper);

    // control
    // --freeze
    adjust_freeze_layers_wrapper = new QHBoxLayout();
    freeze_layers_label = new QLabel("Freeze: ");
    freeze_layers_adjuster = new QSpinBox();
    adjust_freeze_layers_wrapper->addWidget(freeze_layers_label);
    adjust_freeze_layers_wrapper->addWidget(freeze_layers_adjuster);

    // --epochs
    adjust_epochs_wrapper = new QHBoxLayout();
    epochs_label = new QLabel("Epochs: ");
    epochs_adjuster = new QSpinBox();
    epochs_adjuster->setRange(1, 1000);
    epochs_adjuster->setValue(100);
    epochs_adjuster->setSingleStep(50);
    adjust_epochs_wrapper->addWidget(epochs_label);
    adjust_epochs_wrapper->addWidget(epochs_adjuster);

    // --patience
    adjust_patience_epochs_wrapper = new QHBoxLayout();
    patience_epochs_label = new QLabel("Patience: ");
    patience_epochs_adjuster = new QSpinBox();
    patience_epochs_adjuster->setRange(1, 500);
    patience_epochs_adjuster->setValue(50);
    patience_epochs_adjuster->setSingleStep(10);
    adjust_patience_epochs_wrapper->addWidget(patience_epochs_label);
    adjust_patience_epochs_wrapper->addWidget(patience_epochs_adjuster);

    // --batch
    adjust_batch_size_wrapper = new QHBoxLayout();
    batch_size_label = new QLabel("Batch");
    batch_size_adjuster = new QSpinBox();
    batch_size_adjuster->setRange(1, 100);
    batch_size_adjuster->setValue(50);
    batch_size_adjuster->setSingleStep(5);
    adjust_batch_size_wrapper->addWidget(batch_size_label);
    adjust_batch_size_wrapper->addWidget(batch_size_adjuster);

    // --imgsz
    select_image_size_wrapper = new QHBoxLayout();
    image_size_label = new QLabel("Imgsz: ");
    image_size_combobox = new QComboBox();
    image_size_combobox->addItems({ "320", "640" });
    select_image_size_wrapper->addWidget(image_size_label);
    select_image_size_wrapper->addWidget(image_size_combobox);

    // --save
    check_save_wrapper = new QHBoxLayout();
    save_label = new QLabel("Save");
    save_checkbox = new QCheckBox("save");
    save_checkbox->setChecked(train_cfg_opts->control_cfg.save);
    check_save_wrapper->addWidget(save_label);
    check_save_wrapper->addWidget(save_checkbox);

    // --save_period
    adjust_save_period_wrapper = new QHBoxLayout();
    save_period_label = new QLabel("Save_period: ");
    save_period_adjuster = new QSpinBox();
    save_period_adjuster->setRange(1, 100);
    save_period_adjuster->setValue(50);
    save_period_adjuster->setSingleStep(5);
    adjust_save_period_wrapper->addWidget(save_period_label);
    adjust_save_period_wrapper->addWidget(save_period_adjuster);

    control_cfg_layout_wrapper->addLayout(adjust_freeze_layers_wrapper);    // freeze
    control_cfg_layout_wrapper->addLayout(adjust_epochs_wrapper);           // epochs
    control_cfg_layout_wrapper->addLayout(adjust_patience_epochs_wrapper);  // patience
    control_cfg_layout_wrapper->addLayout(adjust_batch_size_wrapper);       // batch
    control_cfg_layout_wrapper->addLayout(select_image_size_wrapper);       // imgsz
    control_cfg_layout_wrapper->addLayout(check_save_wrapper);              // save
    control_cfg_layout_wrapper->addLayout(adjust_save_period_wrapper);      // save_period

    control_cfg_box->setLayout(control_cfg_layout_wrapper);

    basic_and_control_wrapper->addWidget(basic_cfg_box, 1);
    basic_and_control_wrapper->addWidget(control_cfg_box, 1);

    // row_2 col_1
    // advanced
    // --warmup_epochs
    adjust_warmup_epochs_wrapper = new QHBoxLayout();
    warmup_epochs_label = new QLabel("Warmup_epochs");
    warmup_epochs_adjuster = new QSpinBox();
    warmup_epochs_adjuster->setRange(1, 100);
    warmup_epochs_adjuster->setValue(50);
    adjust_warmup_epochs_wrapper->addWidget(warmup_epochs_label);
    adjust_warmup_epochs_wrapper->addWidget(warmup_epochs_adjuster);

    // --warmup_momemntum
    adjust_warmup_momentum_wrapper = new QHBoxLayout();
    warmup_momentum_label = new QLabel("Warmup_momentum");
    warmup_momentum_adjuster = new QSpinBox();
    warmup_momentum_adjuster->setRange(0, 100);
    warmup_momentum_adjuster->setValue(80);
    adjust_warmup_momentum_wrapper->addWidget(warmup_momentum_label);
    adjust_warmup_momentum_wrapper->addWidget(warmup_momentum_adjuster);

    // --warmup_bias_lr
    adjust_warmup_bias_lr_wrapper = new QHBoxLayout();
    warmup_bias_lr_label = new QLabel("Warmup_bias_lr");
    warmup_bias_lr_adjuster = new QSpinBox();
    warmup_bias_lr_adjuster->setRange(1, 100);
    warmup_bias_lr_adjuster->setValue(10);
    adjust_warmup_bias_lr_wrapper->addWidget(warmup_bias_lr_label);
    adjust_warmup_bias_lr_wrapper->addWidget(warmup_bias_lr_adjuster);

    // --verbose
    check_verbose_wrapper = new QHBoxLayout();
    verbose_label = new QLabel("Verbose");
    verbose_checkbox = new QCheckBox("verbose");
    check_verbose_wrapper->addWidget(verbose_label);
    check_verbose_wrapper->addWidget(verbose_checkbox);

    // --seed
    set_seed_wrapper = new QHBoxLayout();
    seed_label = new QLabel("Seed");
    setted_seed = new QLineEdit("");
    setted_seed->setText("seed");
    set_seed_wrapper->addWidget(seed_label);
    set_seed_wrapper->addWidget(setted_seed);

    // --deterministic
    check_deterministic_wrapper = new QHBoxLayout();
    deterministic_label = new QLabel("Deterministic");
    deterministic_checkbox = new QCheckBox();
    check_deterministic_wrapper->addWidget(deterministic_label);
    check_deterministic_wrapper->addWidget(deterministic_checkbox);

    // --single_cls
    check_single_cls_wrapper = new QHBoxLayout();
    single_cls_label = new QLabel("Single_cls");
    single_cls_checkbox = new QCheckBox();
    check_single_cls_wrapper->addWidget(single_cls_label);
    check_single_cls_wrapper->addWidget(single_cls_checkbox);

    // --rect
    check_rect_wrapper = new QHBoxLayout();
    rect_label = new QLabel("Rect");
    rect_checkbox = new QCheckBox();
    check_rect_wrapper->addWidget(rect_label);
    check_rect_wrapper->addWidget(rect_checkbox);

    // --cos_lr
    check_cos_lr_wrapper = new QHBoxLayout();
    cos_lr_label = new QLabel("Cos_lr");
    cos_lr_checkbox = new QCheckBox();
    check_cos_lr_wrapper->addWidget(cos_lr_label);
    check_cos_lr_wrapper->addWidget(cos_lr_checkbox);

    // --close_mosaic
    adjust_close_mosaic_wrapper = new QHBoxLayout();
    close_mosaic_label = new QLabel("Close_mosaic");
    close_mosaic_adjuster = new QSpinBox();
    close_mosaic_adjuster->setRange(1, 100);
    close_mosaic_adjuster->setValue(50);
    adjust_close_mosaic_wrapper->addWidget(close_mosaic_label);
    adjust_close_mosaic_wrapper->addWidget(close_mosaic_adjuster);

    // --resume
    check_resume_wrapper = new QHBoxLayout();
    resume_label = new QLabel("Resume");
    resume_checkbox = new QCheckBox();
    check_resume_wrapper->addWidget(resume_label);
    check_resume_wrapper->addWidget(resume_checkbox);

    // --profile
    check_profile_wrapper = new QHBoxLayout();
    profile_label = new QLabel("Profile");
    profile_checkbox = new QCheckBox();
    check_profile_wrapper->addWidget(profile_label);
    check_profile_wrapper->addWidget(profile_checkbox);

    advanced_cfg_layout_wrapper_col_1->addLayout(adjust_warmup_epochs_wrapper);       // warmup_epochs
    advanced_cfg_layout_wrapper_col_1->addLayout(adjust_warmup_momentum_wrapper);     // warmup_momentum
    advanced_cfg_layout_wrapper_col_1->addLayout(adjust_warmup_bias_lr_wrapper);      // warmup_bias_lr
    advanced_cfg_layout_wrapper_col_1->addLayout(check_verbose_wrapper);              // verbose
    advanced_cfg_layout_wrapper_col_1->addLayout(set_seed_wrapper);                   // seed
    advanced_cfg_layout_wrapper_col_1->addLayout(check_deterministic_wrapper);        // deterministic
    advanced_cfg_layout_wrapper_col_1->addLayout(check_single_cls_wrapper);           // single_cls
    advanced_cfg_layout_wrapper_col_1->addLayout(check_rect_wrapper);                 // rect
    advanced_cfg_layout_wrapper_col_1->addLayout(check_cos_lr_wrapper);               // cos_lr
    advanced_cfg_layout_wrapper_col_1->addLayout(adjust_close_mosaic_wrapper);        // close_mosaic
    advanced_cfg_layout_wrapper_col_1->addLayout(check_resume_wrapper);               // resume
    advanced_cfg_layout_wrapper_col_1->addLayout(check_profile_wrapper);              // profile

    advanced_cfg_box->setLayout(advanced_cfg_layout_wrapper_col_1);

    // row_2_col_2
    // optimizer
    // --optimizer_type
    select_optimizer_type_wrapper = new QHBoxLayout();
    optimizer_type_label = new QLabel("Optimizer_type");
    optimizer_type_combobox = new QComboBox();
    optimizer_type_combobox->addItem("SGD", QVariant::fromValue(Optimizer_Type::SGD));
    optimizer_type_combobox->addItem("Adam", QVariant::fromValue(Optimizer_Type::Adam));
    optimizer_type_combobox->addItem("Adamax", QVariant::fromValue(Optimizer_Type::Adamax));
    optimizer_type_combobox->addItem("NAdam", QVariant::fromValue(Optimizer_Type::NAdam));
    optimizer_type_combobox->addItem("RAdam", QVariant::fromValue(Optimizer_Type::RAdam));
    optimizer_type_combobox->addItem("RMSProp", QVariant::fromValue(Optimizer_Type::RMSProp));
    optimizer_type_combobox->addItem("Auto", QVariant::fromValue(Optimizer_Type::Auto));
    select_optimizer_type_wrapper->addWidget(optimizer_type_label);
    select_optimizer_type_wrapper->addWidget(optimizer_type_combobox);

    // --amp
    check_amp_wrapper = new QHBoxLayout();
    amp_label = new QLabel("Amp");
    amp_checkbox = new QCheckBox();
    check_amp_wrapper->addWidget(amp_label);
    check_amp_wrapper->addWidget(amp_checkbox);

    // --lr0
    adjust_lr0_wrapper = new QHBoxLayout();
    lr0_label = new QLabel("Lr0");
    lr0_adjuster = new QSpinBox();
    lr0_adjuster->setRange(1, 100);
    lr0_adjuster->setValue(1);
    adjust_lr0_wrapper->addWidget(lr0_label);
    adjust_lr0_wrapper->addWidget(lr0_adjuster);

    // --lrf
    adjust_lrf_wrapper = new QHBoxLayout();
    lrf_label = new QLabel("Lrf");
    lrf_adjuster = new QSpinBox();
    lrf_adjuster->setRange(1, 100);
    lrf_adjuster->setValue(1);
    adjust_lrf_wrapper->addWidget(lrf_label);
    adjust_lrf_wrapper->addWidget(lrf_adjuster);

    // --momentum
    adjust_momentum_wrapper = new QHBoxLayout();
    momentum_label = new QLabel("Momentum");
    momentum_adjuster = new QSpinBox();
    momentum_adjuster->setRange(1, 1000);
    momentum_adjuster->setValue(937);
    adjust_momentum_wrapper->addWidget(momentum_label);
    adjust_momentum_wrapper->addWidget(momentum_adjuster);

    // --weight_decay
    adjust_weight_decay_wrapper = new QHBoxLayout();
    weight_decay_label = new QLabel("Weight_decay");
    weight_decay_adjuster = new QSpinBox();
    weight_decay_adjuster->setRange(1, 10000);
    weight_decay_adjuster->setValue(5);
    adjust_weight_decay_wrapper->addWidget(weight_decay_label);
    adjust_weight_decay_wrapper->addWidget(weight_decay_adjuster);

    optimizer_cfg_layout_wrapper->addLayout(select_optimizer_type_wrapper);     // optimizer_type
    optimizer_cfg_layout_wrapper->addLayout(check_amp_wrapper);                 // amp
    optimizer_cfg_layout_wrapper->addLayout(adjust_lr0_wrapper);                // lr0
    optimizer_cfg_layout_wrapper->addLayout(adjust_lrf_wrapper);                // lrf
    optimizer_cfg_layout_wrapper->addLayout(adjust_momentum_wrapper);           // momentum
    optimizer_cfg_layout_wrapper->addLayout(adjust_weight_decay_wrapper);       // weight_decay

    optimizer_cfg_box->setLayout(optimizer_cfg_layout_wrapper);
    optimizer_wrapper->addWidget(optimizer_cfg_box);


    // device
    // --device_type
    select_device_type_wrapper = new QHBoxLayout();
    device_type_label = new QLabel("Device_type");
    device_type_combobox = new QComboBox();
    device_type_combobox->addItem("CPU", QVariant::fromValue(Device_Type::CPU));
    device_type_combobox->addItem("GPU", QVariant::fromValue(Device_Type::GPU));
    device_type_combobox->addItem("CUDA", QVariant::fromValue(Device_Type::CUDA));
    device_type_combobox->addItem("RomC", QVariant::fromValue(Device_Type::RomC));
    select_device_type_wrapper->addWidget(device_type_label);
    select_device_type_wrapper->addWidget(device_type_combobox);
    // --pretrained
    check_pretrained_wrapper = new QHBoxLayout();
    pretrained_label = new QLabel("Pretrained");
    pretrained_checkbox = new QCheckBox();
    check_pretrained_wrapper->addWidget(pretrained_label);
    check_pretrained_wrapper->addWidget(pretrained_checkbox);

    device_cfg_layout_wrapper->addLayout(select_device_type_wrapper);           // device_type
    device_cfg_layout_wrapper->addLayout(check_pretrained_wrapper);             // pretrained

    device_cfg_box->setLayout(device_cfg_layout_wrapper);
    device_and_strategy_wrapper->addWidget(device_cfg_box);

    // strategy
    // --overlap_mask
    check_overlap_mask_wrapper = new QHBoxLayout();
    overlap_mask_label = new QLabel("Overlap_mask");
    overlap_mask_checkbox = new QCheckBox();
    check_overlap_mask_wrapper->addWidget(overlap_mask_label);
    check_overlap_mask_wrapper->addWidget(overlap_mask_checkbox);
    
    // --mask_ratio_and_dropout_adjuster_wrapper
    mask_ratio_and_dropout_adjuster_wrapper = new QHBoxLayout();
    // --mask_ratio
    adjust_mask_ratio_wrapper = new QHBoxLayout();
    mask_ratio_label = new QLabel("Mask_ratio");
    mask_ratio_adjuster = new QSpinBox();
    mask_ratio_adjuster->setRange(1, 10);
    mask_ratio_adjuster->setValue(4);
    adjust_mask_ratio_wrapper->addWidget(mask_ratio_label);
    adjust_mask_ratio_wrapper->addWidget(mask_ratio_adjuster);

    // --dropout
    adjust_dropout_wrapper = new QHBoxLayout();
    dropout_label = new QLabel("Dropout");
    dropout_adjuster = new QSpinBox();
    dropout_adjuster->setRange(0, 100);
    dropout_adjuster->setValue(0);
    adjust_dropout_wrapper->addWidget(dropout_label);
    adjust_dropout_wrapper->addWidget(dropout_adjuster);

    mask_ratio_and_dropout_adjuster_wrapper->addLayout(adjust_mask_ratio_wrapper);
    mask_ratio_and_dropout_adjuster_wrapper->addLayout(adjust_dropout_wrapper);

    strategy_cfg_layout_wrapper->addLayout(check_overlap_mask_wrapper);         // overlap_mask
    strategy_cfg_layout_wrapper->addLayout(mask_ratio_and_dropout_adjuster_wrapper);          // dropout

    strategy_cfg_box->setLayout(strategy_cfg_layout_wrapper);
    device_and_strategy_wrapper->addWidget(strategy_cfg_box);

    optimizer_and_device_and_strategy_wrapper_col_2->addLayout(optimizer_wrapper);
    optimizer_and_device_and_strategy_wrapper_col_2->addLayout(device_and_strategy_wrapper);

    col_2_wrapper->addWidget(advanced_cfg_box, 1);
    col_2_wrapper->addLayout(optimizer_and_device_and_strategy_wrapper_col_2, 1);


    // row_4 col_2
    // --hsv_h
    adjust_hsv_h_label = new QLabel("hsv_h");
    adjust_hsv_h_adjuster = new QSpinBox();
    adjust_hsv_h_wrapper = new QHBoxLayout();
    adjust_hsv_h_adjuster->setRange(0, 1000);
    adjust_hsv_h_adjuster->setValue(15);
    adjust_hsv_h_wrapper->addWidget(adjust_hsv_h_label);
    adjust_hsv_h_wrapper->addWidget(adjust_hsv_h_adjuster);
    // --hsv_s
    adjust_hsv_s_label = new QLabel("hsv_s");
    adjust_hsv_s_adjuster = new QSpinBox();
    adjust_hsv_s_wrapper = new QHBoxLayout();
    adjust_hsv_s_adjuster->setRange(0, 1000);
    adjust_hsv_s_adjuster->setValue(700);
    adjust_hsv_s_wrapper->addWidget(adjust_hsv_s_label);
    adjust_hsv_s_wrapper->addWidget(adjust_hsv_s_adjuster);
    // --hsv_v
    adjust_hsv_v_label = new QLabel("hsv_v");
    adjust_hsv_v_adjuster = new QSpinBox();
    adjust_hsv_v_wrapper = new QHBoxLayout();
    adjust_hsv_v_adjuster->setRange(0, 1000);
    adjust_hsv_v_adjuster->setValue(400);
    adjust_hsv_v_wrapper->addWidget(adjust_hsv_v_label);
    adjust_hsv_v_wrapper->addWidget(adjust_hsv_v_adjuster);
    // --degrees
    adjust_degrees_label = new QLabel("degrees");
    adjust_degrees_adjuster = new QSpinBox();
    adjust_degrees_wrapper = new QHBoxLayout();
    adjust_degrees_adjuster->setRange(0, 1000);
    adjust_degrees_adjuster->setValue(0);
    adjust_degrees_wrapper->addWidget(adjust_degrees_label);
    adjust_degrees_wrapper->addWidget(adjust_degrees_adjuster);
    // --translate
    adjust_translate_label = new QLabel("translate");
    adjust_translate_adjuster = new QSpinBox();
    adjust_translate_wrapper = new QHBoxLayout();
    adjust_translate_adjuster->setRange(0, 1000);
    adjust_translate_adjuster->setValue(100);
    adjust_translate_wrapper->addWidget(adjust_translate_label);
    adjust_translate_wrapper->addWidget(adjust_translate_adjuster);
    // --scale
    adjust_scale_label = new QLabel("scale");
    adjust_scale_adjuster = new QSpinBox();
    adjust_scale_wrapper = new QHBoxLayout();
    adjust_scale_adjuster->setRange(0, 1000);
    adjust_scale_adjuster->setValue(500);
    adjust_scale_wrapper->addWidget(adjust_scale_label);
    adjust_scale_wrapper->addWidget(adjust_scale_adjuster);
    
    augmentation_cfg_layout_col_1_wrapper->addLayout(adjust_hsv_h_wrapper);         // hsv_h
    augmentation_cfg_layout_col_1_wrapper->addLayout(adjust_hsv_s_wrapper);         // hsv_s
    augmentation_cfg_layout_col_1_wrapper->addLayout(adjust_hsv_v_wrapper);         // hsv_v
    augmentation_cfg_layout_col_1_wrapper->addLayout(adjust_degrees_wrapper);       // degrees
    augmentation_cfg_layout_col_1_wrapper->addLayout(adjust_translate_wrapper);     // translate
    augmentation_cfg_layout_col_1_wrapper->addLayout(adjust_scale_wrapper);         // scale

    // --shear
    adjust_shear_label = new QLabel("shear");
    adjust_shear_adjuster = new QSpinBox();
    adjust_shear_wrapper = new QHBoxLayout();
    adjust_shear_adjuster->setRange(0, 1000);
    adjust_shear_adjuster->setValue(0);
    adjust_shear_wrapper->addWidget(adjust_shear_label);
    adjust_shear_wrapper->addWidget(adjust_shear_adjuster);
    // --perspective
    adjust_perspective_label = new QLabel("perspective");
    adjust_perspective_adjuster = new QSpinBox();
    adjust_perspective_wrapper = new QHBoxLayout();
    adjust_perspective_adjuster->setRange(0, 1000);
    adjust_perspective_adjuster->setValue(0);
    adjust_perspective_wrapper->addWidget(adjust_perspective_label);
    adjust_perspective_wrapper->addWidget(adjust_perspective_adjuster);
    // --flipud
    adjust_flipud_label = new QLabel("flipud");
    adjust_flipud_adjuster = new QSpinBox();
    adjust_flipud_wrapper = new QHBoxLayout();
    adjust_flipud_adjuster->setRange(0, 1000);
    adjust_flipud_adjuster->setValue(0);
    adjust_flipud_wrapper->addWidget(adjust_flipud_label);
    adjust_flipud_wrapper->addWidget(adjust_flipud_adjuster);
    // --fliplr
    adjust_fliplr_label = new QLabel("fliplr");
    adjust_fliplr_adjuster = new QSpinBox();
    adjust_fliplr_wrapper = new QHBoxLayout();
    adjust_fliplr_adjuster->setRange(0, 1000);
    adjust_fliplr_adjuster->setValue(1000);
    adjust_fliplr_wrapper->addWidget(adjust_fliplr_label);
    adjust_fliplr_wrapper->addWidget(adjust_fliplr_adjuster);
    // --mosaic
    adjust_mosaic_label = new QLabel("mosaic");
    adjust_mosaic_adjuster = new QSpinBox();
    adjust_mosaic_wrapper = new QHBoxLayout();
    adjust_mosaic_adjuster->setRange(0, 1000);
    adjust_mosaic_adjuster->setValue(0);
    adjust_mosaic_wrapper->addWidget(adjust_mosaic_label);
    adjust_mosaic_wrapper->addWidget(adjust_mosaic_adjuster);
    // --mixup
    adjust_mixup_label = new QLabel("mixup");
    adjust_mixup_adjuster = new QSpinBox();
    adjust_mixup_wrapper = new QHBoxLayout();
    adjust_mixup_adjuster->setRange(0, 1000);
    adjust_mixup_adjuster->setValue(0);
    adjust_mixup_wrapper->addWidget(adjust_mixup_label);
    adjust_mixup_wrapper->addWidget(adjust_mixup_adjuster);
    // --copy_paste
    adjust_copy_paste_label = new QLabel("copy_paste");
    adjust_copy_paste_adjuster = new QSpinBox();
    adjust_copy_paste_wrapper = new QHBoxLayout();
    adjust_copy_paste_adjuster->setRange(0, 1000);
    adjust_copy_paste_adjuster->setValue(15);
    adjust_copy_paste_wrapper->addWidget(adjust_copy_paste_label);
    adjust_copy_paste_wrapper->addWidget(adjust_copy_paste_adjuster);

    
    augmentation_cfg_layout_col_2_wrapper->addLayout(adjust_shear_wrapper);         // shear
    augmentation_cfg_layout_col_2_wrapper->addLayout(adjust_perspective_wrapper);   // prospective
    augmentation_cfg_layout_col_2_wrapper->addLayout(adjust_flipud_wrapper);        // flipud
    augmentation_cfg_layout_col_2_wrapper->addLayout(adjust_fliplr_wrapper);        // fliplr
    augmentation_cfg_layout_col_2_wrapper->addLayout(adjust_mosaic_wrapper);        // mosaic
    augmentation_cfg_layout_col_2_wrapper->addLayout(adjust_mixup_wrapper);         // mixup

    augmentation_cfg_layout_wrapper->addLayout(augmentation_cfg_layout_col_1_wrapper);
    augmentation_cfg_layout_wrapper->addLayout(augmentation_cfg_layout_col_2_wrapper);
    
    QVBoxLayout* test = new QVBoxLayout();
    
    export_cfg_button = new QPushButton("Export");
    reset_cfg_button = new QPushButton("Reset");
    export_and_reset_button_layout_wrapper->addWidget(export_cfg_button);
    export_and_reset_button_layout_wrapper->addWidget(reset_cfg_button);

    test->addLayout(augmentation_cfg_layout_wrapper);
    test->addLayout(adjust_copy_paste_wrapper);
    test->addLayout(export_and_reset_button_layout_wrapper);

    augmentation_cfg_box->setLayout(test);
    row_3_wrapper->addLayout(extra_and_loss_wrapper, 1);
    row_3_wrapper->addWidget(augmentation_cfg_box, 1);


    // row_4 col_1
    // --box
    adjust_box_wrapper = new QHBoxLayout();
    box_label = new QLabel("Box");
    box_adjuster = new QSpinBox();
    box_adjuster->setRange(1, 100);
    box_adjuster->setValue(75);
    adjust_box_wrapper->addWidget(box_label);
    adjust_box_wrapper->addWidget(box_adjuster);

    // --cls
    adjust_cls_wrapper = new QHBoxLayout();
    cls_label = new QLabel("Cls");
    cls_adjuster = new QSpinBox();
    cls_adjuster->setRange(1, 1000);
    cls_adjuster->setValue(5);
    adjust_cls_wrapper->addWidget(cls_label);
    adjust_cls_wrapper->addWidget(cls_adjuster);

    // --dfl
    adjust_dfl_wrapper = new QHBoxLayout();
    dfl_label = new QLabel("Dfl");
    dfl_adjuster = new QSpinBox();
    dfl_adjuster->setRange(1, 1000);
    dfl_adjuster->setValue(15);
    adjust_dfl_wrapper->addWidget(dfl_label);
    adjust_dfl_wrapper->addWidget(dfl_adjuster);
    // --pose
    adjust_pose_wrapper = new QHBoxLayout();
    pose_label = new QLabel("Pose");
    pose_adjuster = new QSpinBox();
    pose_adjuster->setRange(1, 1000);
    pose_adjuster->setValue(120);
    adjust_pose_wrapper->addWidget(pose_label);
    adjust_pose_wrapper->addWidget(pose_adjuster);
    // --kobj
    adjust_kobj_wrapper = new QHBoxLayout();
    kobj_label = new QLabel("Kobj");
    kobj_adjuster = new QSpinBox();
    kobj_adjuster->setRange(1, 1000);
    kobj_adjuster->setValue(10);
    adjust_kobj_wrapper->addWidget(kobj_label);
    adjust_kobj_wrapper->addWidget(kobj_adjuster);
    // --label_smoothing
    adjust_label_smoothing_wrapper = new QHBoxLayout();
    adjust_smoothing_label = new QLabel("Label_smoothing");
    label_smoothing_adjuster = new QSpinBox();
    label_smoothing_adjuster->setRange(1, 1000);
    label_smoothing_adjuster->setValue(0);
    adjust_label_smoothing_wrapper->addWidget(adjust_smoothing_label);
    adjust_label_smoothing_wrapper->addWidget(label_smoothing_adjuster);

    loss_cfg_layout_wrapper->addLayout(adjust_box_wrapper);
    loss_cfg_layout_wrapper->addLayout(adjust_cls_wrapper);
    loss_cfg_layout_wrapper->addLayout(adjust_dfl_wrapper);
    loss_cfg_layout_wrapper->addLayout(adjust_pose_wrapper);
    loss_cfg_layout_wrapper->addLayout(adjust_kobj_wrapper);
    loss_cfg_layout_wrapper->addLayout(adjust_label_smoothing_wrapper);

    loss_cfg_box->setLayout(loss_cfg_layout_wrapper);


    select_nbs_wrapper = new QHBoxLayout();
    select_nbs_label = new QLabel("nbs");
    select_nbs_combobox = new QComboBox();
    select_nbs_combobox->addItems({"4", "8", "16", "32", "64"});
    select_nbs_wrapper->addWidget(select_nbs_label);
    select_nbs_wrapper->addWidget(select_nbs_combobox);

    extra_cfg_layout_wrapper->addLayout(select_nbs_wrapper);

    extra_cfg_box->setLayout(extra_cfg_layout_wrapper);

    extra_and_loss_wrapper->addWidget(loss_cfg_box);
    extra_and_loss_wrapper->addWidget(extra_cfg_box);

    config_panel->addLayout(basic_and_control_wrapper);
    config_panel->addLayout(col_2_wrapper);
    config_panel->addLayout(row_3_wrapper);
    config_panel->addLayout(config_panel_wrapper);
    setLayout(config_panel);

    // config_slots
    // basic
    // --data
    connect( select_data_file_button
           , &QPushButton::clicked, this
           , [this] {
                QString yaml_data_filepath = 
                    QFileDialog::getOpenFileName( this, "Select yaml file"
                                                , "", "YAML Files (*.yaml);;ALL Files (*)"
                                                ) ;
                if (!yaml_data_filepath.isEmpty()) {
                    train_cfg_opts->basic_cfg.data = yaml_data_filepath;
                    QFileInfo file_info(yaml_data_filepath);
                    selected_data_file_label->setText("Data: " + file_info.fileName());
                    selected_data_file_label->setToolTip(yaml_data_filepath);
                    DEBUG_LOG("Selected Yaml data: " << yaml_data_filepath);
                }
           }) ;
    // --cache
    connect( cache_type_combobox
           , QOverload<int>::of(&QComboBox::currentIndexChanged)
           , this
           , [this](int idx) {
                Cache_Type selected_cache_type =
                    cache_type_combobox->itemData(idx).value<Cache_Type>();
                train_cfg_opts->basic_cfg.cache = selected_cache_type; 
                DEBUG_LOG("Selected Cache Type: " << Enum_Reflect<Cache_Type>::to_string(selected_cache_type) << "Index: " << idx);

            }) ;
    // --workers
    connect( workers_num_combobox
           , QOverload<int>::of(&QComboBox::currentIndexChanged)
           , this
           , [this](int idx) {
                auto& workers = train_cfg_opts->basic_cfg.workers;
                workers = 1 << idx;
                DEBUG_LOG("Selected Workers Num: " << workers); 
            }) ;
    // --project
    connect( setted_project_alias
           , &QLineEdit::textChanged
           , this
           , [this](const QString& alias) {
                train_cfg_opts->basic_cfg.project = alias; 
                DEBUG_LOG("Project Alias: " << alias);
            }) ;
    // --name
    connect( setted_name_alias
           , &QLineEdit::textChanged
           , this
           , [this](const QString& alias) {
                train_cfg_opts->basic_cfg.name = alias;\
                DEBUG_LOG("Name Alias: " << alias);
            }) ;
    // --exist_ok
    connect( exist_ok_checkbox
           , &QCheckBox::toggled
           , this
           , [this](bool checked) {
                train_cfg_opts->basic_cfg.exist_ok = checked;
                DEBUG_LOG("Exist_Ok: " << (checked ? "True" : "False")); 
            }) ;
    // --fraction
    connect( fraction_adjuster
           , &QSlider::valueChanged
           , this
           , [this](int frac) {
                auto& fraction = train_cfg_opts->basic_cfg.fraction;
                fraction = frac / 100.f;
                fraction_label->setText("Fraction: " + QString::number(fraction, 'f', 2));
                DEBUG_LOG("Fraction: " << fraction); 
            }) ;

    // control
    // --freeze
    connect( freeze_layers_adjuster
           , QOverload<int>::of(&QSpinBox::valueChanged)
           , this
           , [this](int freeze_layers){
                train_cfg_opts->control_cfg.freeze = freeze_layers;
                DEBUG_LOG("Freeze: " << freeze_layers); 
            }) ;
    // --epochs
    connect( epochs_adjuster
           , QOverload<int>::of(&QSpinBox::valueChanged)
           , this
           , [this](int epochs){
                train_cfg_opts->control_cfg.epochs = epochs;
                patience_epochs_adjuster->setMaximum(epochs);
                if (patience_epochs_adjuster->value() > epochs)
                    patience_epochs_adjuster->setValue(epochs / 2);
                DEBUG_LOG("Epochs: " << epochs); 
            }) ;
    // --patience
    connect( patience_epochs_adjuster
           , QOverload<int>::of(&QSpinBox::valueChanged)
           , this
           , [this](int patience) {
                train_cfg_opts->control_cfg.patience = patience;
                DEBUG_LOG("Patience: " << patience); 
            }) ;
    // --batch
    connect( batch_size_adjuster
           , QOverload<int>::of(&QSpinBox::valueChanged)
           , this
           , [this](int batch_size){
                train_cfg_opts->control_cfg.batch = batch_size;
                DEBUG_LOG("Batch_Size: " << batch_size);
           });
    // --imgsz
    connect ( image_size_combobox
            , QOverload<int>::of(&QComboBox::currentIndexChanged)
            , this
            , [this](int idx) {
                int imgsz = image_size_combobox->itemText(idx).toInt();
                train_cfg_opts->control_cfg.imgsz = imgsz;
                DEBUG_LOG("Selected image size: " << imgsz);
            });
    // --save
    connect ( save_checkbox
            , &QCheckBox::toggled, this
            , [this](bool checked) {
                train_cfg_opts->control_cfg.save = checked;
                DEBUG_LOG("Check save: " << (checked ? "true" : "false"));
            });
    // --save_period
    connect (save_period_adjuster
            , QOverload<int>::of(&QSpinBox::valueChanged)
            , this
            , [this](int save_period) {
                train_cfg_opts->control_cfg.save_period = save_period;
                DEBUG_LOG("Selected save period: " << save_period);
                train_cfg_opts->show_cfg();
            });
    // device
    // --device
    connect( device_type_combobox
           , QOverload<int>::of(&QComboBox::currentIndexChanged)
           , this
           , [this](int idx) {
                Device_Type selected_device_type =
                    device_type_combobox->itemData(idx).value<Device_Type>();
                train_cfg_opts->device_cfg.device = selected_device_type; 
                DEBUG_LOG("Selected Device Type: " << Enum_Reflect<Device_Type>::to_string(selected_device_type) << "Index: " << idx);
            });
    // --pretrained
    connect( pretrained_checkbox
           , &QCheckBox::toggled, this
           , [this](bool checked) {
                train_cfg_opts->device_cfg.pretrained = checked;
                DEBUG_LOG("Check pretrained: " << (checked ? "true" : "false"));
            });
    // advanced
    // --warmup_epochs
    connect (warmup_epochs_adjuster
            , QOverload<int>::of(&QSpinBox::valueChanged)
            , this
            , [this](int value) {
                train_cfg_opts->advanced_cfg.warmup_epochs = value;
                DEBUG_LOG("Selected warmup epochs: " << value);
            });
    // --warmup_momentum
    connect ( warmup_momentum_adjuster
            , QOverload<int>::of(&QSpinBox::valueChanged)
            , this
            , [this](int value) {
                train_cfg_opts->advanced_cfg.warmup_momentum = value / 100.f;
                DEBUG_LOG("Selected warmup momentum:" << train_cfg_opts->advanced_cfg.warmup_momentum);
            });
    // --warmup_bias_lr
    connect ( warmup_bias_lr_adjuster
            , QOverload<int>::of(&QSpinBox::valueChanged)
            , this
            , [this](int value){
                train_cfg_opts->advanced_cfg.warmup_bias_lr = value / 100.f;
                DEBUG_LOG("Selected warmup bias lr:" << train_cfg_opts->advanced_cfg.warmup_bias_lr);
            });
    // --verbose
    connect ( verbose_checkbox
            , &QCheckBox::toggled, this
            , [this](bool checked) {
                train_cfg_opts->advanced_cfg.verbose = checked;
                DEBUG_LOG("Check verbose: " << (checked ? "true" : "false"));
            });
    // --seed
    connect (setted_seed
            , &QLineEdit::textChanged
            , this
            , [this](const QString& text){
                const int seed = text.toInt();
                train_cfg_opts->advanced_cfg.seed = seed;
                DEBUG_LOG("Setted seed:" << seed);
            });
    // --deterministic
    connect (deterministic_checkbox
            , &QCheckBox::toggled
            , this
            , [this](bool checked){
                train_cfg_opts->advanced_cfg.deterministic = checked;
                DEBUG_LOG("Check deterministic:" << (checked ? "true" : "false"));
            });
    // --single_cls
    connect (single_cls_checkbox
            , &QCheckBox::toggled
            , this
            , [this](bool checked){
                train_cfg_opts->advanced_cfg.single_cls = checked;
                DEBUG_LOG("Check single_cls: " << (checked ? "true" : "false"));
            });
    // --rect
    connect ( rect_checkbox
            , &QCheckBox::toggled
            , this
            , [this](bool checked){
                train_cfg_opts->advanced_cfg.rect = checked;
                DEBUG_LOG("Check rect: " << (checked ? "true" : "false"));
            });
    // --cos_lr
    connect ( cos_lr_checkbox
            , &QCheckBox::toggled
            , this
            , [this](bool checked){
                train_cfg_opts->advanced_cfg.cos_lr = checked;
                DEBUG_LOG("Check cos_lr:" << (checked ? "true" : "false"));
            });
    // --close_mosaic
    connect (close_mosaic_adjuster
            , QOverload<int>::of(&QSpinBox::valueChanged)
            , this
            , [this](int epochs){
                train_cfg_opts->advanced_cfg.close_mosaic = epochs;
                DEBUG_LOG("Selected closs_mosaic:" << epochs);
            });
    // --resume
    connect ( resume_checkbox
            , &QCheckBox::toggled
            , this
            , [this](bool checked){
                train_cfg_opts->advanced_cfg.resume = checked;
                DEBUG_LOG("Check resume: " << (checked ? "true" : "false"));
            });
    // --profile
    connect (profile_checkbox
            , &QCheckBox::toggled
            , this
            , [this](bool checked) {
                train_cfg_opts->advanced_cfg.profile =checked;
                DEBUG_LOG("Check profile: " << (checked ? "true" : "false"));
            });
    // optimizer
    // --optimizer_type
    connect( optimizer_type_combobox
        , QOverload<int>::of(&QComboBox::currentIndexChanged)
        , this
        , [this](int idx) {
             Optimizer_Type selected_optimizer_type =
                 optimizer_type_combobox->itemData(idx).value<Optimizer_Type>();
             train_cfg_opts->optimizer_cfg.optimizer = selected_optimizer_type; 
             DEBUG_LOG("Selected optimizer type: " << Enum_Reflect<Optimizer_Type>::to_string(selected_optimizer_type) << "Index: " << idx);
         });
    // --amp
    connect (amp_checkbox
            , &QCheckBox::toggled
            , this
            , [this](bool checked){
                train_cfg_opts->optimizer_cfg.amp = checked;
                DEBUG_LOG("Check amp: " << (checked ? "true" : "false"));
            });
    // --lr0
    connect (lr0_adjuster
            , QOverload<int>::of(&QSpinBox::valueChanged)
            , this
            , [this](int value){
                train_cfg_opts->optimizer_cfg.lr0 = value / 100.f;
                DEBUG_LOG("Selected lr0:" << train_cfg_opts->optimizer_cfg.lr0); 
            });
    // --lrf
    connect (lrf_adjuster
            , QOverload<int>::of(&QSpinBox::valueChanged)
            , this
            , [this](int value){
                train_cfg_opts->optimizer_cfg.lrf = value / 100.f;
                DEBUG_LOG("Selected lrf:" << train_cfg_opts->optimizer_cfg.lrf);
            });
    // --momentum
    connect ( momentum_adjuster
            , QOverload<int>::of(&QSpinBox::valueChanged)
            , this
            , [this](int value) {
                train_cfg_opts->optimizer_cfg.momentum = value / 1000.f;
                DEBUG_LOG("Selected momentum: " << train_cfg_opts->optimizer_cfg.momentum);
            });
    // --weight_decay
    connect ( weight_decay_adjuster
            , QOverload<int>::of(&QSpinBox::valueChanged)
            , this
            , [this](int value){
                train_cfg_opts->optimizer_cfg.weight_decay = value / 10000.f;
                DEBUG_LOG("Selected weight_decay:" << train_cfg_opts->optimizer_cfg.weight_decay);
            });
    // strategy
    // --overlap_mask
    connect ( overlap_mask_checkbox
            , &QCheckBox::toggled
            , this
            , [this](bool checked) {
                train_cfg_opts->strategy_cfg.segamentation_cfg.overlap_mask = checked;
                DEBUG_LOG("Check Segamentation: overlap_mask:" << (checked ? "true" : "false"));
            });
    // --mask_ratio
    connect ( mask_ratio_adjuster
            , QOverload<int>::of(&QSpinBox::valueChanged)
            , this
            , [this](int value){
                train_cfg_opts->strategy_cfg.segamentation_cfg.mask_ratio = value;
                DEBUG_LOG("Selected mask_ratio: " << value);
            });
    // --dropout
    connect (dropout_adjuster
            , QOverload<int>::of(&QSpinBox::valueChanged)
            , this
            , [this](int value) {
                train_cfg_opts->strategy_cfg.classifcation_cfg.dropout = value / 100.f;
                DEBUG_LOG("Selected dropout: " << train_cfg_opts->strategy_cfg.classifcation_cfg.dropout);
            });
    // augmentation
    // --hsv_h
    connect (adjust_hsv_h_adjuster
            , QOverload<int>::of(&QSpinBox::valueChanged)
            , this
            , [this](int value){
                train_cfg_opts->augmentation_cfg.hsv_h = value / 1000.f;
                DEBUG_LOG("Selected hsv_h:" << train_cfg_opts->augmentation_cfg.hsv_h);
            });
    // --hsv_s
    connect (adjust_hsv_s_adjuster
        , QOverload<int>::of(&QSpinBox::valueChanged)
        , this
        , [this](int value){
            train_cfg_opts->augmentation_cfg.hsv_s = value / 1000.f;
            DEBUG_LOG("Selected hsv_s:" << train_cfg_opts->augmentation_cfg.hsv_s);
        });
    // --hsv_v
    connect (adjust_hsv_v_adjuster
        , QOverload<int>::of(&QSpinBox::valueChanged)
        , this
        , [this](int value){
            train_cfg_opts->augmentation_cfg.hsv_v = value / 1000.f;
            DEBUG_LOG("Selected hsv_v:" << train_cfg_opts->augmentation_cfg.hsv_v);
        });
    // --degrees
    connect (adjust_degrees_adjuster
        , QOverload<int>::of(&QSpinBox::valueChanged)
        , this
        , [this](int value){
            train_cfg_opts->augmentation_cfg.degrees = value / 1000.f;
            DEBUG_LOG("Selected degrees:" << train_cfg_opts->augmentation_cfg.degrees);
        });
    // --translate
    connect (adjust_translate_adjuster
        , QOverload<int>::of(&QSpinBox::valueChanged)
        , this
        , [this](int value){
            train_cfg_opts->augmentation_cfg.translate = value / 1000.f;
            DEBUG_LOG("Selected translate:" << train_cfg_opts->augmentation_cfg.translate);
        });
    // --scale
    connect (adjust_scale_adjuster
        , QOverload<int>::of(&QSpinBox::valueChanged)
        , this
        , [this](int value){
            train_cfg_opts->augmentation_cfg.scale = value / 1000.f;
            DEBUG_LOG("Selected scale:" << train_cfg_opts->augmentation_cfg.scale);
        });
    // --shear
    connect (adjust_shear_adjuster
        , QOverload<int>::of(&QSpinBox::valueChanged)
        , this
        , [this](int value){
            train_cfg_opts->augmentation_cfg.shear = value / 1000.f;
            DEBUG_LOG("Selected shear:" << train_cfg_opts->augmentation_cfg.shear);
        });
    // --perspective
    connect (adjust_perspective_adjuster
        , QOverload<int>::of(&QSpinBox::valueChanged)
        , this
        , [this](int value){
            train_cfg_opts->augmentation_cfg.perspective = value / 1000.f;
            DEBUG_LOG("Selected perspective:" << train_cfg_opts->augmentation_cfg.perspective);
        });
    // --filpud
    connect (adjust_flipud_adjuster
        , QOverload<int>::of(&QSpinBox::valueChanged)
        , this
        , [this](int value){
            train_cfg_opts->augmentation_cfg.flipud = value / 1000.f;
            DEBUG_LOG("Selected flipud:" << train_cfg_opts->augmentation_cfg.flipud);
        });
    // --fliplr
    connect (adjust_fliplr_adjuster
        , QOverload<int>::of(&QSpinBox::valueChanged)
        , this
        , [this](int value){
            train_cfg_opts->augmentation_cfg.fliplr = value / 1000.f;
            DEBUG_LOG("Selected fliplr:" << train_cfg_opts->augmentation_cfg.fliplr);
        });
    // --mosaic
    connect (adjust_mosaic_adjuster
        , QOverload<int>::of(&QSpinBox::valueChanged)
        , this
        , [this](int value){
            train_cfg_opts->augmentation_cfg.mosaic = value / 1000.f;
            DEBUG_LOG("Selected mosaic:" << train_cfg_opts->augmentation_cfg.mosaic);
        });
    // --mixup
    connect (adjust_mixup_adjuster
        , QOverload<int>::of(&QSpinBox::valueChanged)
        , this
        , [this](int value){
            train_cfg_opts->augmentation_cfg.mixup = value / 1000.f;
            DEBUG_LOG("Selected mixup:" << train_cfg_opts->augmentation_cfg.mixup);
        });
    // --copy_paste
    connect (adjust_copy_paste_adjuster
        , QOverload<int>::of(&QSpinBox::valueChanged)
        , this
        , [this](int value){
            train_cfg_opts->augmentation_cfg.copy_paste = value / 1000.f;
            DEBUG_LOG("Selected copy_paste:" << train_cfg_opts->augmentation_cfg.copy_paste);
        });
    
    // loss
    // --box
    connect (box_adjuster
        , QOverload<int>::of(&QSpinBox::valueChanged)
        , this
        , [this](int value){
            train_cfg_opts->loss_cfg.box = value / 10.f;
            DEBUG_LOG("Selected box:" << train_cfg_opts->loss_cfg.box);
        });
    // --cls
    connect (cls_adjuster
        , QOverload<int>::of(&QSpinBox::valueChanged)
        , this
        , [this](int value){
            train_cfg_opts->loss_cfg.cls = value / 10.f;
            DEBUG_LOG("Selected cls:" << train_cfg_opts->loss_cfg.cls);
        });
    // --dfl
    connect (dfl_adjuster
        , QOverload<int>::of(&QSpinBox::valueChanged)
        , this
        , [this](int value){
            train_cfg_opts->loss_cfg.dfl = value / 10.f;
            DEBUG_LOG("Selected dfl:" << train_cfg_opts->loss_cfg.dfl);
        });
    // --pose
    connect (pose_adjuster
        , QOverload<int>::of(&QSpinBox::valueChanged)
        , this
        , [this](int value){
            train_cfg_opts->loss_cfg.pose = value / 10.f;
            DEBUG_LOG("Selected pose:" << train_cfg_opts->loss_cfg.pose);
        });
    // --kobj
    connect (kobj_adjuster
        , QOverload<int>::of(&QSpinBox::valueChanged)
        , this
        , [this](int value){
            train_cfg_opts->loss_cfg.kobj = value / 10.f;
            DEBUG_LOG("Selected kobj:" << train_cfg_opts->loss_cfg.kobj);
        });
    // --label_smoothing
    connect (label_smoothing_adjuster
        , QOverload<int>::of(&QSpinBox::valueChanged)
        , this
        , [this](int value){
            train_cfg_opts->loss_cfg.label_smoothing = value / 10.f;
            DEBUG_LOG("Selected label_smoothing:" << train_cfg_opts->loss_cfg.label_smoothing);
        });
    
    // extra
    // --nbs
    connect ( select_nbs_combobox
        , QOverload<int>::of(&QComboBox::currentIndexChanged)
        , this
        , [this](int idx) {
            int nbs = select_nbs_combobox->itemText(idx).toInt();
            train_cfg_opts->extra_cfg.nbs = nbs;
            DEBUG_LOG("Selected nbs: " << nbs);
        });

    // export button
    connect (export_cfg_button, &QPushButton::clicked, this, &Train_Configuration_Panel::export_cfg);
}

#include <fstream>
void Train_Configuration_Panel::export_cfg() {
    QString file_path 
        = QFileDialog::getSaveFileName( this, "export config"
                                      , QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)
                                      , "config file (*.json)"
                                      ) ;
    if (file_path.isEmpty()) return;

    try {
        nlohmann::json j;
        train_cfg_opts->to_json(j);

        j["metadata"] = {
            {"exportime", QDateTime::currentDateTime().toString(Qt::ISODate).toStdString()},
            {"app_version", "1.0.0"}
        };

        std::ofstream file(file_path.toStdString());
        if (!file.is_open())
            throw std::runtime_error("Failed to create config file");
        file << j.dump(2);
        
    } catch (const std::exception& e) {

    }
}



Train_Configuration_Panel::~Train_Configuration_Panel() {

}






















