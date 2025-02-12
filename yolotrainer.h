#ifndef YOLOTRAINER_H
#define YOLOTRAINER_H

#include <QObject>
#include <boost/process.hpp>
#include <boost/process/v1/environment.hpp>
#include <fstream>
#include <string>
#include <memory>
#include <QDateTime>
#include <filesystem>
#include <string>
#include <atomic>


struct training_log_data {
    int epoch;  
};

class yoloTrainer: public QObject{
    Q_OBJECT
public:
    explicit yoloTrainer(QObject* parent = nullptr);
    ~yoloTrainer();
    std::filesystem::path workDir;


    void stopTraining();
signals:
    void logMessage(const QString& message);
    void trainingFinished(int exitCode);
    void training_progress(int epoch, float mAP);

public slots:
    void runTraining();

private:

    std::filesystem::path scriptPath;
    std::filesystem::path python_exe;
    std::filesystem::path logFilePath;

    std::atomic<bool> stopFlag{false};
    std::mutex log_mutex;

    boost::process::child c;
    std::unique_ptr<std::thread> outThread;
    std::unique_ptr<std::thread> errThread;

    void terminateProcessTree(boost::process::pid_t pid);
    void setup_process_environment(boost::process::environment& env);
};

#endif // YOLOTRAINER_H

