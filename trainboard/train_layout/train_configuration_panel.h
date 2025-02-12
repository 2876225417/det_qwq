#ifndef TRAIN_CONFIGURATION_PANEL_H
#define TRAIN_CONFIGURATION_PANEL_H

#include <QString>
#include <nlohmann/detail/macro_scope.hpp>
#include <qboxlayout.h>
#include <qbrush.h>
#include <qcombobox.h>
#include <qdebug.h>
#include <qlayout.h>
#include <qpushbutton.h>
#include <qscopedpointer.h>
#include <qspinbox.h>

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>


namespace train_config_type {
    enum class Cache_Type {
        None,
        RAM,
        Disk,
    };

    enum class Device_Type {
        CPU,
        GPU,
        CUDA,
        RomC,
        Other,   
    };

    enum class Optimizer_Type {
        SGD,
        Adam,
        Adamax,
        AdamW,
        NAdam,
        RAdam,
        RMSProp,
        Auto,
    };

    template <typename Enum>
    struct Enum_Reflect;

    template <>
    struct Enum_Reflect<Cache_Type> {
        static constexpr const char* to_string(Cache_Type value) {
            switch (value) {
                case Cache_Type::None:  return "None";
                case Cache_Type::RAM:   return "RAM";
                case Cache_Type::Disk:  return "Disk";
                default: return "Unknown";
            }
        }

        static constexpr
        int size() { return 3; }

        static constexpr
        const char* types[] = { "None", "RAM", "Disk" };
    };

    template <>
    struct Enum_Reflect<Device_Type> {
        static constexpr const char* to_string(Device_Type value) {
            switch (value) {
                case Device_Type::CPU:      return "CPU";
                case Device_Type::GPU:      return "GPU";
                case Device_Type::CUDA:     return "CUDA";
                case Device_Type::RomC:     return "RomC";
                case Device_Type::Other:    return "Other";
                default:                    return "Unknown";
            }
        }

        static constexpr
        int size() { return 5; };

        static constexpr
        const char* types[] = { "CPU", "GPU", "CUDA", "RomC", "Other" };
    };

    template <>
    struct Enum_Reflect<Optimizer_Type> {
        static constexpr const char* to_string(Optimizer_Type value) {
            switch (value) {
            case Optimizer_Type::SGD:       return "SGD";
            case Optimizer_Type::Adam:      return "Adam";
            case Optimizer_Type::Adamax:    return "Adamax";
            case Optimizer_Type::AdamW:     return "AdamW";
            case Optimizer_Type::NAdam:     return "NAdam";
            case Optimizer_Type::RAdam:     return "RAdam";
            case Optimizer_Type::RMSProp:   return "RMSProp";
            case Optimizer_Type::Auto:      return "Auto";
            default:                        return "Unknown";
            }
        }

        static constexpr
        int size() { return 8; }

        static constexpr
        const char* types[] = { "SGD"
                              , "Adam"
                              , "Adamax"
                              , "AdamW"
                              , "NAdam"
                              , "RAdam"
                              , "RMSProp"
                              , "Auto"
                              } ;
    };


}

using namespace train_config_type;
struct train_config_options{
    // 基本
    struct {
        QString         data;                                   // 设置训练数据配置文件
        Cache_Type      cache           = Cache_Type::RAM;      // 指定使用缓存加载数据的设备类型
        int             workers         = 8;                    // 指定数据加载的工作线程数
        QString         project         = "runs/train";         // 设置项目名称
        QString         name            = "exp";                // 设置实验名称exp(结构:prject/name)
        bool            exist_ok        = false;                // 是否覆盖现有实验exp
        float           fraction        = 1.f;                  // 指定要训练的数据集百分比(默认1.0，为训练集中的所有图像)
    } basic_cfg;

    // 训练控制
    struct {
        int             freeze          = 0;                    // 指定在训练期间冻结的前 n 层
        int             epochs          = 100;                  // 设置训练周期数
        int             patience        = 50;                   // 指定等待无明显改善以进行早期停止的周期数
        int             batch           = 16;                   // 指定每批次的图像数量(-1为自动批处理)
        int             imgsz           = 640;                  // 指定输入图像的大小(一般为640)
        bool            save            = true;                 // 是否保存训练检查点和预测结果
        int             save_period     = -1;                   // 设置保存检查点的周期(小于1则禁用)
    } control_cfg;

    // 设备
    struct {
        Device_Type     device          = Device_Type::CUDA;    // 指定使用的设备
        bool            pretrained      = true;                 // 是否使用预训练模型
    } device_cfg;

    // 优化器
    struct {
        Optimizer_Type  optimizer       = Optimizer_Type::SGD;  // 选择优化器类型
        bool            amp             = true;                 // 是否启动自动混合精度(AMP)训练
        float           lr0             = 0.01f;                // 指定初始学习率
        float           lrf             = 0.01f;                // 指定最终学习率
        float           momentum        = 0.937;                // SGD动量/Adam beta1
        float           weight_decay    = 0.0005;               // 指定优化器权重衰减
    } optimizer_cfg;

    // 高级训练配置
    struct {
        float           warmup_epochs   = 3.f;                  // 指定预热周期数
        float           warmup_momentum = 0.8f;                 // 指定预热初始动量
        float           warmup_bias_lr  = 0.1f;                 // 指定预热初始偏执学习率
        bool            verbose         = true;                 // 是否打印详细输出
        int             seed            = 0;                    // 设置可重复性的随机种子
        bool            deterministic   = true;                 // 是否启用确定性模式
        bool            single_cls      = false;                // 将多类数据训练为单类
        bool            rect            = false;                // 如果 mode == "train" 则进行矩形训练, 如果 mode == "val" 则进行矩形验证
        bool            cos_lr          = false;                // 是否使用余弦学习率调度器
        int             close_mosaic    = 0;                    // 指定最后几个周期禁用马赛克增强
        bool            resume          = false;                // 是否从上个检查点恢复训练
        bool            profile         = false;                // 是否在训练期间为记录器启用 ONNX 和 TensorRT 速度
    } advanced_cfg;

    // 训练策略
    struct {
        struct {    // 分割
            bool        overlap_mask    = true;                 // 分割训练：是否在训练期间启用重叠掩码
            int         mask_ratio      = 4;                    // 分割训练：指定掩码降采样比例
        } segamentation_cfg;
        struct {    // 分类
            float       dropout         = 0.f;                  // 分类训练：指定丢弃正则化比例
        } classifcation_cfg;
    } strategy_cfg;

    // 数据增强
    struct {
        float           hsv_h           = 0.015f;               // 指定图像 HSV-Hue 增强分数
        float           hsv_s           = 0.7f;                 // 指定图像 HSV-Saturation 增强分数
        float           hsv_v           = 0.4f;                 // 指定图像 HSV-Value 增强分数
        float           degrees         = 0.f;                  // 指定图像旋转度数 (+/- deg)
        float           translate       = 0.1f;                 // 指定图像平移分数 (+/- 比例)
        float           scale           = 0.5f;                 // 指定图像缩放比例 (+/- 比例)
        float           shear           = 0.f;                  // 指定图像裁剪比例 (+/- 比例)
        float           perspective     = 0.f;                  // 指定图像透视分数 (+/- 比例，范围：0-0.001)
        float           flipud          = 0.f;                  // 指定图像上下翻转概率
        float           fliplr          = 0.5f;                 // 指定图像左右翻转概率
        float           mosaic          = 1.f;                  // 指定对图像添加马赛克效果的概率
        float           mixup           = 0.f;                  // 指定图像混合的概率
        float           copy_paste      = 0.f;                  // 指定图像分割复制/粘贴的概率
    } augmentation_cfg;

    // 损失增益
    struct {
        float           box             = 7.5f;                 // 指定盒损失增益
        float           cls             = 0.5f;                 // 指定类别损失增益
        float           dfl             = 1.5f;                 // 指定dfl损失增益
        float           pose            = 12.f;                 // 指定姿势损失增益
        float           kobj            = 1.f;                  // 指定关键点对象损失增益
        float           label_smoothing = 0.f;                  // 指定标签平滑分数
    } loss_cfg;

    // 其他
    struct {
        int             nbs             = 64;                   // 指定名义批量大小
    } extra_cfg;


    NLOHMANN_JSON_SERIALIZE_ENUM(Device_Type, {
        {Device_Type::CPU,          "CPU"},
        {Device_Type::GPU,          "GPU"},
        {Device_Type::CUDA,         "CUDA"},
        {Device_Type::RomC,         "Romc"},
        {Device_Type::Other,        "Other"}
    });

    NLOHMANN_JSON_SERIALIZE_ENUM(Cache_Type,{
        {Cache_Type::None,          "None"},
        {Cache_Type::RAM,           "RAM"},
        {Cache_Type::Disk,          "Disk"}
    });

    NLOHMANN_JSON_SERIALIZE_ENUM(Optimizer_Type, {
        {Optimizer_Type::SGD,       "SGD"},
        {Optimizer_Type::Adam,      "Adam"},
        {Optimizer_Type::Adamax,    "Adamax"},
        {Optimizer_Type::AdamW,     "AdamW"},
        {Optimizer_Type::NAdam,     "NAdam"},
        {Optimizer_Type::RAdam,     "RAdam"},
        {Optimizer_Type::RMSProp,   "RMSProp"},
        {Optimizer_Type::Auto,      "Auto"}
    });
    
    void to_json(nlohmann::json& j) {
        j["basic"] = {
            {"data",            basic_cfg.data.toStdString()},
            {"cache",           Enum_Reflect<Cache_Type>::to_string(basic_cfg.cache)},
            {"workers",         basic_cfg.workers},
            {"project",         basic_cfg.project.toStdString()},
            {"name",            basic_cfg.name.toStdString()},
            {"exist_ok",        basic_cfg.exist_ok},
            {"fraction",        basic_cfg.fraction}
        };

        j["control"] = {
            {"freeze",          control_cfg.freeze},
            {"epochs",          control_cfg.epochs},
            {"patience",        control_cfg.patience},
            {"batch",           control_cfg.batch},
            {"imgsz",           control_cfg.imgsz},
            {"save",            control_cfg.save},
            {"save_period",     control_cfg.save_period}
        };

        j["device"] = {
            {"type",            Enum_Reflect<Device_Type>::to_string(device_cfg.device)},
            {"pretrained",      device_cfg.pretrained}
        };

        j["optimizer"] = {
            {"optimizer",       Enum_Reflect<Optimizer_Type>::to_string(optimizer_cfg.optimizer)},
            {"amp",             optimizer_cfg.amp},
            {"lr0",             optimizer_cfg.lr0},
            {"lrf",             optimizer_cfg.lrf},
            {"momentum",        optimizer_cfg.momentum},
            {"weight_decay",    optimizer_cfg.weight_decay}
        };

        j["advanced"] = {
            {"warmup_epochs",   advanced_cfg.warmup_epochs},
            {"warmup_momentum", advanced_cfg.warmup_momentum},
            {"warmup_bias_lr",  advanced_cfg.warmup_bias_lr},
            {"verbose",         advanced_cfg.verbose},
            {"seed",            advanced_cfg.seed},
            {"deterministic",   advanced_cfg.deterministic},
            {"single_cls",      advanced_cfg.single_cls},
            {"rect",            advanced_cfg.rect},
            {"cos_lr",          advanced_cfg.cos_lr},
            {"close_mosaic",    advanced_cfg.close_mosaic},
            {"resume",         advanced_cfg.resume},
            {"profile",        advanced_cfg.profile},
        };

        j["strategy"] = {
            {"overlap_mask",    strategy_cfg.segamentation_cfg.overlap_mask},
            {"mask_ratio",      strategy_cfg.segamentation_cfg.mask_ratio},
            {"dropout",         strategy_cfg.classifcation_cfg.dropout}
        };

        j["augmentation"] = {
            {"hsv_h",           augmentation_cfg.hsv_h},
            {"hsv_s",           augmentation_cfg.hsv_s},
            {"hsv_v",           augmentation_cfg.hsv_v},
            {"degrees",         augmentation_cfg.degrees},
            {"translate",       augmentation_cfg.translate},
            {"scale",           augmentation_cfg.scale},
            {"shear",           augmentation_cfg.shear},
            {"perspective",     augmentation_cfg.perspective},
            {"flipud",          augmentation_cfg.flipud},
            {"fliplr",          augmentation_cfg.fliplr},
            {"mosaic",         augmentation_cfg.mosaic},
            {"mixup",          augmentation_cfg.mixup},
            {"copy_paste",     augmentation_cfg.copy_paste}   
        };

        j["loss"] = {
            {"box",             loss_cfg.box},
            {"cls",             loss_cfg.cls},
            {"dfl",             loss_cfg.dfl},
            {"pose",            loss_cfg.pose},
            {"kobj",            loss_cfg.kobj},
            {"label_smoothing", loss_cfg.label_smoothing}
        };
        
        j["extra"] = {
            {"nbs", extra_cfg.nbs}
        };
    }

    explicit
    train_config_options() { }
    explicit
    train_config_options(const QString& data_file)
        : basic_cfg{data_file} { }

    void show_cfg( bool basic = true, bool controlling = true
                 , bool device = true, bool optimizer = true
                 , bool advanced = true, bool strategy = true
                 , bool augmentation = true, bool loss = true
                 , bool extra = true ) const {
        qDebug().noquote() << "\n[Training Configurations]";
        if (basic) {
            qDebug() << "\n[Basic Config]";
            qDebug() << " Data:" << basic_cfg.data;
            qDebug() << " Cache_Type:" << Enum_Reflect<Cache_Type>::to_string(basic_cfg.cache);
            qDebug() << " Workers:" << basic_cfg.workers;
            qDebug() << " Project:" << basic_cfg.project;
            qDebug() << " Name:" << basic_cfg.name;
            qDebug() << " Exist_ok:" << (basic_cfg.exist_ok ? "true" : "false");
            qDebug() << " Fraction:" << basic_cfg.fraction;
            qDebug() << "-----------------------------------";
        }

        if (controlling) {
            qDebug() << "\n[Controlling Config]";
            qDebug() << " Freeze:" << control_cfg.freeze;
            qDebug() << " Epochs:" << control_cfg.epochs;
            qDebug() << " Patience:" << control_cfg.patience;
            qDebug() << " Batch:" << control_cfg.batch;
            qDebug() << " Image Size:" << control_cfg.imgsz;
            qDebug() << " Save:" << (control_cfg.save ? "true" : "false");
            qDebug() << " Save Period:" << control_cfg.save_period;
            qDebug() << "----------------------------------";
        }

        if (device) {
            qDebug() << "\n[Device Config]";
            qDebug() << " Device_Type:" << Enum_Reflect<Device_Type>::to_string(device_cfg.device);
            qDebug() << " Pretrained:" << device_cfg.pretrained;
            qDebug() << "----------------------------------";
        }

        if (optimizer) {
            qDebug() << "\n[Optimizer Config]";
            qDebug() << " Optimizer_Type:" << Enum_Reflect<Optimizer_Type>::to_string(optimizer_cfg.optimizer);
            qDebug() << " AMP:" << (optimizer_cfg.amp ? "true" : "false");
            qDebug() << " lr0:" << optimizer_cfg.lr0;
            qDebug() << " lrf:" << optimizer_cfg.lrf;
            qDebug() << " Momentum:" << optimizer_cfg.momentum;
            qDebug() << " Weight_Decay:" << optimizer_cfg.weight_decay;
            qDebug() << "---------------------------------";
        }

        if (advanced) {
            qDebug() << "\n[Advanced Config]";
            qDebug() << " Warmup_Epochs:" << advanced_cfg.warmup_epochs;
            qDebug() << " Warmup_Momentum:" << advanced_cfg.warmup_momentum;
            qDebug() << " Warmup_Bias_lr:" << advanced_cfg.warmup_bias_lr;
            qDebug() << " Verbose:" << (advanced_cfg.verbose ? "true" : "false");
            qDebug() << " Seed:" << advanced_cfg.seed;
            qDebug() << " Deterministic:" << (advanced_cfg.deterministic ? "true" : "false");
            qDebug() << " Single_cls:" << (advanced_cfg.single_cls ? "true" : "false");
            qDebug() << " Rect:" << (advanced_cfg.rect ? "true" : "false");
            qDebug() << " Cos_lr:" << (advanced_cfg.cos_lr ? "true" : "false");
            qDebug() << " Close_mosaic:" << advanced_cfg.close_mosaic;
            qDebug() << " Resume:" << (advanced_cfg.resume ? "true" : "false");
            qDebug() << " Profile:" << (advanced_cfg.profile ? "true" : "false");
            qDebug() << "--------------------------------";
        }

        if (strategy) {
            qDebug() << "\n[Strategy Config]";
            qDebug() << " [Segamentation]";
            qDebug() << "  Overlap_Mask:" << (strategy_cfg.segamentation_cfg.overlap_mask ? "true" : "false");
            qDebug() << "  Mask_Ratio:" << strategy_cfg.segamentation_cfg.mask_ratio;
            qDebug() << " [Classification]";
            qDebug() << "  Dropout:" << strategy_cfg.classifcation_cfg.dropout;
            qDebug() << "--------------------------------";
        }

        if (augmentation) {
            qDebug() << "\n[Augmentation Config]";
            qDebug() << " HSV_H:" << augmentation_cfg.hsv_h;
            qDebug() << " HSV_S:" << augmentation_cfg.hsv_s;
            qDebug() << " HSV_V:" << augmentation_cfg.hsv_v;
            qDebug() << " Degrees:" << augmentation_cfg.degrees;
            qDebug() << " Translate:" << augmentation_cfg.translate;
            qDebug() << " Scale:" << augmentation_cfg.scale;
            qDebug() << " Shear:" << augmentation_cfg.shear;
            qDebug() << " Perspective:" << augmentation_cfg.perspective;
            qDebug() << " Filpud:" << augmentation_cfg.flipud;
            qDebug() << " Filplr:" << augmentation_cfg.fliplr;
            qDebug() << " Mosaic:" << augmentation_cfg.mosaic;
            qDebug() << " Mixup:" << augmentation_cfg.mixup;
            qDebug() << " Copy_Paste:" << augmentation_cfg.copy_paste;
            qDebug() << "-------------------------------";
        }

        if (loss) {
            qDebug() << "\n[Loss Config]";
            qDebug() << " Box:" << loss_cfg.box;
            qDebug() << " Cls:" << loss_cfg.cls;
            qDebug() << " Dfl:" << loss_cfg.dfl;
            qDebug() << " Pose:" << loss_cfg.pose;
            qDebug() << " Kobj:" << loss_cfg.kobj;
            qDebug() << " Label_Smoothing:" << loss_cfg.label_smoothing;
            qDebug() << "-------------------------------";
        }

        if (extra) {
            qDebug() << "\n[Extra Config]";
            qDebug() << " Nbs:" << extra_cfg.nbs;
            qDebug() << "-------------------------------";

        }
    }
};


#include <exception>

class invalid_config_exception: public std::exception {
public:
    explicit
    invalid_config_exception(const QString& msg)
        : msg_(msg.toStdString()) { }

    explicit
    invalid_config_exception(QString&& msg)
        : msg_(std::move(msg).toStdString()) { }

    const char*
    what() const noexcept override {
        return msg_.c_str();
    }

private:
    std::string msg_;

};


#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>

#include <QLabel>
#include <QSpinBox>
#include <QLineEdit>
#include <QCheckBox>
#include <QComboBox>
#include <QSlider>
#include <QPushButton>
#include <QFileDialog>
#include <QDebug>
#include <QGroupBox>
#include <QToolTip>
#include <QSpinBox>

#include <memory>

extern "C" void highLoad(int* result, int N);

#define fixed_file_label "Data: "


#define DEBUG_LOG(msg) qDebug() << msg; 

#ifdef TRAINBOARD_DEBUG
        // basic
        #define CACHE_TYPE_LOG(msg)         qDebug() << msg;
        #define WORKERS_NUM_LOG(msg)        qDebug() << msg;
        #define WORKERS_NUM_GET_LOG(msg)    qDebug() << msg;
        #define PROJECT_ALIAS_LOG(msg)      qDebug() << msg;
        #define NAME_ALIAS_LOG(msg)         qDebug() << msg;
        #define EXIST_OK_LOG(msg)           qDebug() << msg;
        #define FRACTION_LOG(msg)           qDebug() << msg;
        // control
        #define FREEZE_LOG(msg)             qDebug() << msg;
        #define EPOCHS_LOG(msg)             qDebug() << msg;
        #define PATIENCE_LOG(msg)           qDebug() << msg;
        #define BATCH_LOG(msg)              qDebug() << msg;
        #define IMGSZ_LOG(msg)              qDebug() << msg;
        #define SAVE_LOG(msg)               qDebug() << msg;
        #define SAVE_PERIOD(msg)            qDebug() << msg;
#else
    // basic
    #define CACHE_TYPE_LOG(msg)
    #define WORKERS_NUM_LOG(msg)
    #define WORKERS_NUM_GET_LOG(msg)
    #define PROJECT_ALIAS_LOG(msg)
    #define NAME_ALIAS_LOG(msg)
    #define EXIST_OK_LOG(msg)
    #define FRACTION_LOG(msg)
    // control
    #define FREEZE_LOG(msg)
    #define EPOCHS_LOG(msg)
    #define PATIENCE_LOG(msg)
    #define BATCH_LOG(msg)
    #define IMGSZ_LOG(msg)
    #define SAVE_LOG(msg)
    #define SAVE_PERIOD(msg)
#endif


class Train_Configuration_Panel: public QWidget {
public:
    explicit Train_Configuration_Panel(QWidget* parent = nullptr);
    ~Train_Configuration_Panel();

private slots:
    void on_select_data_file_button_clicked();

private:
    std::unique_ptr<train_config_options> train_cfg_opts_default;
    std::unique_ptr<train_config_options> train_cfg_opts;

    
    void train_cfg_opts_show() const;

    QVBoxLayout*    config_panel;

    QVBoxLayout*    config_panel_wrapper;

    // row_1
    QHBoxLayout*    basic_and_control_wrapper;
    // basic
    QGroupBox*      basic_cfg_box;
    QVBoxLayout*    basic_cfg_layout_wrapper;
    // --data
    QHBoxLayout*    select_data_file_wrapper;
    QPushButton*    select_data_file_button;
    QLabel*         selected_data_file_label;
    // --cache
    QHBoxLayout*    select_cache_type_wrapper;
    QLabel*         selected_cache_type_label;
    QComboBox*      cache_type_combobox;
    // --workers
    QHBoxLayout*    select_workers_num_wrapper;
    QLabel*         selected_workers_num_label;
    QComboBox*      workers_num_combobox;
    // --project
    QHBoxLayout*    set_project_alias_wrapper;
    QLabel*         project_alias_label;
    QLineEdit*      setted_project_alias;
    // --name
    QHBoxLayout*    set_name_alias_wrapper;
    QLabel*         name_alias_label;
    QLineEdit*      setted_name_alias;
    // --exist_ok
    QHBoxLayout*    check_exist_ok_wrapper;
    QLabel*         exist_ok_label;
    QCheckBox*      exist_ok_checkbox;
    // --fraction
    QHBoxLayout*    adjust_fraction_wrapper;
    QLabel*         fraction_label;
    QSlider*        fraction_adjuster;

    // control
    QGroupBox*      control_cfg_box;
    QVBoxLayout*    control_cfg_layout_wrapper;
    // --freeze
    QHBoxLayout*    adjust_freeze_layers_wrapper;
    QLabel*         freeze_layers_label;
    QSpinBox*       freeze_layers_adjuster;
    // --epochs
    QHBoxLayout*    adjust_epochs_wrapper;
    QLabel*         epochs_label;
    QSpinBox*       epochs_adjuster;
    // --patience
    QHBoxLayout*    adjust_patience_epochs_wrapper;
    QLabel*         patience_epochs_label;
    QSpinBox*       patience_epochs_adjuster;
    // --batch
    QHBoxLayout*    adjust_batch_size_wrapper;
    QLabel*         batch_size_label;
    QSpinBox*       batch_size_adjuster;
    // --imgsz
    QHBoxLayout*    select_image_size_wrapper;
    QLabel*         image_size_label;
    QComboBox*      image_size_combobox;
    // --save
    QHBoxLayout*    check_save_wrapper;
    QLabel*         save_label;
    QCheckBox*      save_checkbox;
    // --save_period
    QHBoxLayout*    adjust_save_period_wrapper;
    QLabel*         save_period_label;
    QSpinBox*       save_period_adjuster;



    // row_2 col_1
    QHBoxLayout*    col_2_wrapper;
    QGroupBox*      advanced_cfg_box;
    QHBoxLayout*    advanced_col_wrapper;
    // advanced
    QVBoxLayout*    advanced_cfg_layout_wrapper_col_1;
    // --warmup_epochs
    QHBoxLayout*    adjust_warmup_epochs_wrapper;
    QLabel*         warmup_epochs_label;
    QSpinBox*       warmup_epochs_adjuster;
    // --warmup_momentum
    QHBoxLayout*    adjust_warmup_momentum_wrapper;
    QLabel*         warmup_momentum_label;
    QSpinBox*       warmup_momentum_adjuster;
    // --warmup_bias_lr
    QHBoxLayout*    adjust_warmup_bias_lr_wrapper;
    QLabel*         warmup_bias_lr_label;
    QSpinBox*       warmup_bias_lr_adjuster;
    // --verbose
    QHBoxLayout*    check_verbose_wrapper;
    QLabel*         verbose_label;
    QCheckBox*      verbose_checkbox;
    // --seed
    QHBoxLayout*    set_seed_wrapper;
    QLabel*         seed_label;
    QLineEdit*      setted_seed;
    // --deterministic
    QHBoxLayout*    check_deterministic_wrapper;
    QLabel*         deterministic_label;
    QCheckBox*      deterministic_checkbox;
    // --single_cls
    QHBoxLayout*    check_single_cls_wrapper;
    QLabel*         single_cls_label;
    QCheckBox*      single_cls_checkbox;
    // --rect
    QHBoxLayout*    check_rect_wrapper;
    QLabel*         rect_label;
    QCheckBox*      rect_checkbox;
    // --cos_lr
    QHBoxLayout*    check_cos_lr_wrapper;
    QLabel*         cos_lr_label;
    QCheckBox*      cos_lr_checkbox;
    // --close_mosaic
    QHBoxLayout*    adjust_close_mosaic_wrapper;
    QLabel*         close_mosaic_label;
    QSpinBox*       close_mosaic_adjuster;
    // --resume
    QHBoxLayout*    check_resume_wrapper;
    QLabel*         resume_label;
    QCheckBox*      resume_checkbox;
    // --profile
    QHBoxLayout*    check_profile_wrapper;
    QLabel*         profile_label;
    QCheckBox*      profile_checkbox;


    // row_2 col_2
    QVBoxLayout*    optimizer_and_device_and_strategy_wrapper_col_2;
    QVBoxLayout*    optimizer_wrapper;
    // optimizer
    QGroupBox*      optimizer_cfg_box;
    QVBoxLayout*    optimizer_cfg_layout_wrapper;
    // --optimizer_type
    QHBoxLayout*    select_optimizer_type_wrapper;
    QLabel*         optimizer_type_label;
    QComboBox*      optimizer_type_combobox;
    // --amp
    QHBoxLayout*    check_amp_wrapper;
    QLabel*         amp_label;
    QCheckBox*      amp_checkbox;
    // --lr0
    QHBoxLayout*    adjust_lr0_wrapper;
    QLabel*         lr0_label;
    QSpinBox*       lr0_adjuster;
    // --lrf
    QHBoxLayout*    adjust_lrf_wrapper;
    QLabel*         lrf_label;
    QSpinBox*       lrf_adjuster;
    // --momentum
    QHBoxLayout*    adjust_momentum_wrapper;
    QLabel*         momentum_label;
    QSpinBox*       momentum_adjuster;
    // --weight_decay
    QHBoxLayout*    adjust_weight_decay_wrapper;
    QLabel*         weight_decay_label;
    QSpinBox*       weight_decay_adjuster;

    QVBoxLayout*    device_and_strategy_wrapper;
    // device
    QGroupBox*      device_cfg_box;
    QVBoxLayout*    device_cfg_layout_wrapper;
    // --device_type
    QHBoxLayout*    select_device_type_wrapper;
    QLabel*         device_type_label;
    QComboBox*      device_type_combobox;
    // --pretrained
    QHBoxLayout*    check_pretrained_wrapper;
    QLabel*         pretrained_label;
    QCheckBox*      pretrained_checkbox;

    // strategy
    QGroupBox*      strategy_cfg_box;
    QVBoxLayout*    strategy_cfg_layout_wrapper;
    // --overlap_mask
    QHBoxLayout*    check_overlap_mask_wrapper;
    QLabel*         overlap_mask_label;
    QCheckBox*      overlap_mask_checkbox;
    // --mask_ratio_and_dropout_adjuster_wrapper;
    QHBoxLayout*    mask_ratio_and_dropout_adjuster_wrapper;
    // --mask_ratio
    QHBoxLayout*    adjust_mask_ratio_wrapper;
    QLabel*         mask_ratio_label;
    QSpinBox*       mask_ratio_adjuster;
    // --dropout
    QHBoxLayout*    adjust_dropout_wrapper;
    QLabel*         dropout_label;
    QSpinBox*       dropout_adjuster;

    // row_3_col_1
    QHBoxLayout*    row_3_wrapper;
    QGroupBox*      augmentation_cfg_box;
    // augmentation
    QHBoxLayout*    augmentation_cfg_layout_wrapper;
    // augmentation col_1
    QVBoxLayout*    augmentation_cfg_layout_col_1_wrapper; 
    // --hsv_h
    QHBoxLayout*    adjust_hsv_h_wrapper;
    QLabel*         adjust_hsv_h_label;
    QSpinBox*       adjust_hsv_h_adjuster;
    // --hsv_s
    QHBoxLayout*    adjust_hsv_s_wrapper;
    QLabel*         adjust_hsv_s_label;
    QSpinBox*       adjust_hsv_s_adjuster;
    // --hsv_v
    QHBoxLayout*    adjust_hsv_v_wrapper;
    QLabel*         adjust_hsv_v_label;
    QSpinBox*       adjust_hsv_v_adjuster;
    // --degrees
    QHBoxLayout*    adjust_degrees_wrapper;
    QLabel*         adjust_degrees_label;
    QSpinBox*       adjust_degrees_adjuster;
    // --translate
    QHBoxLayout*    adjust_translate_wrapper;
    QLabel*         adjust_translate_label;
    QSpinBox*       adjust_translate_adjuster;
    // --scale
    QHBoxLayout*    adjust_scale_wrapper;
    QLabel*         adjust_scale_label;
    QSpinBox*       adjust_scale_adjuster;


    // augmentation col_2
    QVBoxLayout*    augmentation_cfg_layout_col_2_wrapper;
    // --shear
    QHBoxLayout*    adjust_shear_wrapper;
    QLabel*         adjust_shear_label;
    QSpinBox*       adjust_shear_adjuster;
    // --perspective
    QHBoxLayout*    adjust_perspective_wrapper;
    QLabel*         adjust_perspective_label;
    QSpinBox*       adjust_perspective_adjuster;
    // --flipud
    QHBoxLayout*    adjust_flipud_wrapper;
    QLabel*         adjust_flipud_label;
    QSpinBox*       adjust_flipud_adjuster;
    // --fliplr
    QHBoxLayout*    adjust_fliplr_wrapper;
    QLabel*         adjust_fliplr_label;
    QSpinBox*       adjust_fliplr_adjuster;
    // --mosaic
    QHBoxLayout*    adjust_mosaic_wrapper;
    QLabel*         adjust_mosaic_label;
    QSpinBox*       adjust_mosaic_adjuster;
    // --mixup
    QHBoxLayout*    adjust_mixup_wrapper;
    QLabel*         adjust_mixup_label;
    QSpinBox*       adjust_mixup_adjuster;
    // --copy_paste
    QHBoxLayout*    adjust_copy_paste_wrapper;
    QLabel*         adjust_copy_paste_label;
    QSpinBox*       adjust_copy_paste_adjuster;


    
    // 
    QVBoxLayout*    extra_and_loss_wrapper;
    QGroupBox*      extra_cfg_box;
    // loss
    QVBoxLayout*    loss_cfg_layout_wrapper;
    // --box
    QHBoxLayout*    adjust_box_wrapper;
    QLabel*         box_label;
    QSpinBox*       box_adjuster;
    // --cls
    QHBoxLayout*    adjust_cls_wrapper;
    QLabel*         cls_label;
    QSpinBox*       cls_adjuster;
    // --dfl
    QHBoxLayout*    adjust_dfl_wrapper;
    QLabel*         dfl_label;
    QSpinBox*       dfl_adjuster;
    // --pose
    QHBoxLayout*    adjust_pose_wrapper;
    QLabel*         pose_label;
    QSpinBox*       pose_adjuster;
    // --kobj
    QHBoxLayout*    adjust_kobj_wrapper;
    QLabel*         kobj_label;
    QSpinBox*       kobj_adjuster;
    // --label_smoothing
    QHBoxLayout*    adjust_label_smoothing_wrapper;
    QLabel*         adjust_smoothing_label;
    QSpinBox*       label_smoothing_adjuster;

    QGroupBox*      loss_cfg_box;
    // extra
    QVBoxLayout*    extra_cfg_layout_wrapper;
    // --nbs
    QHBoxLayout*    select_nbs_wrapper;
    QLabel*         select_nbs_label;
    QComboBox*      select_nbs_combobox;


    // export and reset
    QHBoxLayout*    export_and_reset_button_layout_wrapper;
    // export
    QPushButton*    export_cfg_button;
    // reset        
    QPushButton*    reset_cfg_button;


    void export_cfg();
    void reset_cfg();
};

#endif // TRAIN_CONFIGURATION_PANEL_H
