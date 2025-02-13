#include "trainboard.h"
#include "utils/gpu_info/gpu_performance_monitor.h"
#include "yolotrainer.h"
#include <qboxlayout.h>
#include <qevent.h>
#include <qfilesystemwatcher.h>
#include <qgroupbox.h>
#include <qiodevicebase.h>
#include <qjsonobject.h>
#include <qlabel.h>
#include <qpoint.h>
#include <qprocess.h>
#include <qprogressbar.h>
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
    
    QHBoxLayout* training_dashboard = new QHBoxLayout();
    QVBoxLayout* control_training_layout_wrapper = new QVBoxLayout();
    QVBoxLayout* training_indicator_layout_wrapper = new QVBoxLayout();

    startButton = new QPushButton("Start Training", this);
    control_training_layout_wrapper->addWidget(startButton);

    stopButton = new QPushButton("Stop Training", this);
    stopButton->setEnabled(false);
    control_training_layout_wrapper->addWidget(stopButton);

    QProgressBar* training_progress = new QProgressBar(this);

    connect ( performance_panel->m_train_log_websocket
            , &get_train_log_websocket::training_metrics
            , this
            , [this, training_progress](QJsonObject json) {
                int epoch = json["epoch"].toInt();
                int total_epochs = json["total_epochs"].toInt();
                qDebug() << "Testting signal";
                qDebug() << "Epoch:" << epoch;
                qDebug() << "Total Epochs:" << total_epochs;
                training_progress->setMaximum(total_epochs);
                training_progress->setValue(epoch);
            });

    training_indicator_layout_wrapper->addWidget(training_progress);
    

    logDisplay = new QPlainTextEdit(this);
    logDisplay->setReadOnly(true);

    
    
    training_dashboard->addLayout(control_training_layout_wrapper);
    training_dashboard->addLayout(training_indicator_layout_wrapper);
    training_console_wrapper->addLayout(training_dashboard);
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
