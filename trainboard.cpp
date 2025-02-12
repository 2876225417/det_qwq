#include "trainboard.h"
#include "utils/gpu_info/gpu_performance_monitor.h"
#include "yolotrainer.h"
#include <qboxlayout.h>
#include <qgroupbox.h>
#include <qlabel.h>
#include <qtextoption.h>
#include <qtimer.h>




TrainBoard::TrainBoard(QWidget* parent)
    : QWidget(parent)
    , gpu_info_panel(new GPU_INFO_PANEL(this))
    , trainer(new yoloTrainer)
    , trainerThread(new QThread(this)) {
    QHBoxLayout* trainboard = new QHBoxLayout();

    QHBoxLayout* train_board_wrapper = new QHBoxLayout();
    

    // 训练监视
    QVBoxLayout* training_log_wrapper = new QVBoxLayout();
    QGroupBox* training_log_box = new QGroupBox("Training Log");
    
    QVBoxLayout* performance_monitor_console_wrapper = new QVBoxLayout();
    QGroupBox* performance_monitor_console = new QGroupBox();


    gpu_performance_monitor* performance_panel = new gpu_performance_monitor();

    performance_monitor_console_wrapper->addWidget(performance_panel);
    performance_monitor_console->setLayout(performance_monitor_console_wrapper);

    QVBoxLayout* training_console_wrapper = new QVBoxLayout();
    QGroupBox* training_console = new QGroupBox("Training Console");
    
    TrainLabel = new QLabel("Training Control Board", this);
    training_console_wrapper->addWidget(TrainLabel);

    startButton = new QPushButton("Start Training", this);
    training_console_wrapper->addWidget(startButton);

    stopButton = new QPushButton("Stop Training", this);
    stopButton->setEnabled(false);
    training_console_wrapper->addWidget(stopButton);

    logDisplay = new QPlainTextEdit(this);
    logDisplay->setReadOnly(true);
    training_console_wrapper->addWidget(logDisplay);
    
    
    training_console->setLayout(training_console_wrapper);
    
    
    training_log_wrapper->addWidget(performance_monitor_console, 7);
    training_log_wrapper->addWidget(training_console, 3);
    training_log_box->setLayout(training_log_wrapper);


    
    // 训练配置
    QGroupBox* training_configuration_box = new QGroupBox("Training Configuration");
    QHBoxLayout* training_configuration_wrapper = new QHBoxLayout();
    Train_Configuration_Panel* train_configuration_panel = new Train_Configuration_Panel();
    training_configuration_wrapper->addWidget(train_configuration_panel);

    training_configuration_box->setLayout(training_configuration_wrapper);

    train_board_wrapper->addWidget(training_log_box, 3);
    train_board_wrapper->addWidget(training_configuration_box, 2);
    

    QLabel* test_1 = new QLabel();
    QLabel* test_2 = new QLabel();
    
    train_board_wrapper->addWidget(test_1);
    train_board_wrapper->addWidget(test_2);

    setLayout(train_board_wrapper);

    


    trainer->moveToThread(trainerThread);

    connect( startButton
           , &QPushButton::clicked
           , this
           , &TrainBoard::startTraining
           ) ;

    connect( trainer
           , &yoloTrainer::logMessage
           , this
           , &TrainBoard::appendLog
           ) ;

    connect( trainer
           , &yoloTrainer::trainingFinished
           , this
           , &TrainBoard::onTrainingFinished
           ) ;

    connect( trainerThread
           , &QThread::finished
           , trainer
           , &QObject::deleteLater
           ) ;

    connect( stopButton
           , &QPushButton::clicked
           , this
           , &TrainBoard::stopTraining
           );

    trainerThread->start();
}

TrainBoard::~TrainBoard(){
    trainerThread->quit();
    trainerThread->wait();
}

void TrainBoard::updateGPUInfo(){

}

void TrainBoard::startTraining(){
    startButton->setEnabled(false);
    stopButton->setEnabled(true);
    qDebug() << "Runned training!";
    std::filesystem::path res_csv;

    QTimer* delay_get = new QTimer(this);
    delay_get->setSingleShot(true);
    connect(delay_get, &QTimer::timeout, this, [&]() {
        qDebug() << "Latest: " << get_latest_result_dir();
        res_csv = get_latest_result_dir() / "results.csv";
    });
    delay_get->start(12000);
    

    QMetaObject::invokeMethod(
        trainer
        , "runTraining"
        , Qt::QueuedConnection
        );
}

void TrainBoard::stopTraining(){
    stopButton->setEnabled(false);
    trainer->stopTraining();

    qDebug() << "stopping button clicked";
}


void TrainBoard::appendLog(const QString& message){
    logDisplay->appendPlainText(message);
}

void TrainBoard::onTrainingFinished(int exitCode){
    startButton->setEnabled(true);
    stopButton->setEnabled(false);
    if(exitCode == 0){
        appendLog("\nTraining completed successfully!");
    } else {
        appendLog("\nTraining failed with exti code: " + QString::number(exitCode));
    }
}

std::filesystem::path TrainBoard::get_latest_result_dir() {
    const auto base_dir = trainer->workDir / "runs" / "train";
    std::filesystem::path latest_dir;
    std::time_t latest_time = 0;

    try {
        for (const auto& entry: std::filesystem::directory_iterator(base_dir)) {
            if (entry.is_directory() &&
                entry.path().filename().string().find("exp") != std::string::npos) {
                
                auto write_time = std::chrono::system_clock::to_time_t(std::chrono::file_clock::to_sys((entry.last_write_time())));

                if (write_time > latest_time) {
                    latest_time = write_time;
                    latest_dir = entry.path();
                }
            }
        }
    } catch(const std::filesystem::filesystem_error& e) {

    }
    return latest_dir;
}