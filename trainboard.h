#ifndef TRAINBOARD_H
#define TRAINBOARD_H

#include <QWidget>
#include <QPushButton>
#include <QPlainTextEdit>
#include <QLabel>
#include <QThread>
#include <QVBoxLayout>
#include <QObject>
#include <QFileDialog>
#include <QMessageBox>
#include <QGroupBox>

#include "yolotrainer.h"
#include "gpuinfo.h"
#include "trainboard/train_layout/train_configuration_panel.h"


class TrainBoard: public QWidget{
    Q_OBJECT
public:
    explicit TrainBoard(QWidget* parent = nullptr);
    ~TrainBoard();

    void updateGPUInfo();

private:
    void startTraining();
    void appendLog(const QString& message);
    void onTrainingFinished(int exitCode);
    void stopTraining();
    void runRealTimeDetection();
    void displayFrame(const QImage& frame);
    void stopReadTimeDetection();
    void _choose_file();
    void _start_detection();
    void _pause_detection();

    std::filesystem::path get_latest_result_dir();
    
    
private:
    // Layout
    QLabel* TrainLabel;
    QLabel* GPUInfoLabel;


    GPU_INFO_PANEL* gpu_info_panel;

    QPushButton* startButton;
    QPushButton* stopButton;
    QPushButton* detectButton;
    QPushButton* realTimeDetectButton;
    QPlainTextEdit* logDisplay;

    QPushButton* _choose_file_button;
    QPushButton* _start_detection_button;
    QPushButton* _pause_detection_button;




    QLabel* _display_label;
    QString _selected_file;


    QLabel* realTimeDetectionLabel;
    QTimer* frameUpdateTimer;

    bool realTimeDetectionActive;

    yoloTrainer* trainer;
    QThread* trainerThread;

    QTimer* csv_reslover;
    

};




#endif // TRAINBOARD_H
