

#include <boost/process/v1/detail/on_exit.hpp>
#include <boost/process/v1/environment.hpp>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#endif

#include "yolotrainer.h"
#include <boost/process.hpp>
#include <config.h>
#include <chrono>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #define PROCESS_HIDE boost::process::windows::hide
#elif defined(__linux__)
    #include <unistd.h>
    #define PROCESS_HIDE
#endif

yoloTrainer::yoloTrainer(QObject* parent):
    QObject(parent),
    workDir(build_cfg::YOLO_SCRIPT_DIR),
    scriptPath(workDir / "train.py"),
#ifdef ENABLE_VENV
    python_exe(build_cfg::PYTHON_VENV_EXE),
#else
    python_exe("python3"),
#endif
    logFilePath(workDir / "training_log.txt") {
    if (!std::filesystem::exists(python_exe)) 
        throw std::runtime_error("Python interpreter path is invalid: " + python_exe.string());

    if (!std::filesystem::exists(scriptPath)) 
        throw std::runtime_error("Training script is invalid: " + scriptPath.string());
}

yoloTrainer::~yoloTrainer(){
    if(c.running()){
        c.terminate();
        c.wait();
    }

    if(outThread && outThread->joinable()) outThread->join();
    if(errThread && errThread->joinable()) errThread->join();
}

void yoloTrainer::setup_process_environment(boost::process::environment& env) {
#ifdef ENABLE_VENV
    const auto venv_bin_path = build_cfg::YOLO_SCRIPT_DIR +     
    #ifdef _WIN32
        "\\Scripts"
    env["path"] += ";" + venv_bin_path;
    #else
        "/bin";
    env["path"] = ":" + venv_bin_path;
    #endif
#endif
    env["PYTHONUNBUFFERED"] = "1";
}


// 开始模型训练
void yoloTrainer::runTraining(){
    try{
        std::lock_guard<std::mutex> lock(log_mutex);
        stopFlag.store(false);

        std::ofstream logFile(
            logFilePath,
            std::ios::out | std::ios::trunc
            );
        if(!logFile.is_open()){
            emit logMessage(
                "Failed to open log file: "
                + QString::fromStdString(logFilePath.string()));
            emit trainingFinished(-1);
            return;
        }

        boost::process::ipstream out_stream;
        boost::process::ipstream err_stream;

        boost::process::environment env = boost::process::environment();
        setup_process_environment(env);
#ifdef _WIN32
        c = boost::process::child(
            "" " + scriptPath.string(),
            boost::process::std_out > out_stream,
            boost::process::std_err > err_stream,
            boost::process::start_dir = workDir.string()
            );
#else
        c = boost::process::child(
            python_exe.string(),
            scriptPath.string(),
            boost::process::std_out > out_stream,
            boost::process::std_err > err_stream,
            boost::process::start_dir = workDir.string(),
            boost::process::env = env,
            boost::process::on_exit([this](int, const std::error_code&){
                emit trainingFinished(c.exit_code());
            })
        );
#endif

        
        outThread = std::make_unique<std::thread>([&](){
            std::string line;
            while(out_stream && std::getline(out_stream, line)){
                if(stopFlag.load()) break;
                emit logMessage(QString("[OUT] ") + QString::fromStdString(line));
                logFile << "[OUT] " << line << '\n';
            }
        });

        errThread = std::make_unique<std::thread>([&](){
            std::string line;
            while(err_stream && std::getline(err_stream, line)){
                if(stopFlag.load()) break;
                emit logMessage(QString("[ERR] ") + QString::fromStdString(line));
                logFile << "[ERR] " << line << '\n';
            }
        });

        while(c.running()){
            if(stopFlag.load()){
                c.terminate();
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if(stopFlag.load()) c.terminate();

        if(outThread->joinable()) outThread->join();
        if(errThread->joinable()) errThread->join();

        emit trainingFinished(c.exit_code());
    }catch(const std::exception& e){
        emit logMessage("Error while calling Python scripts: " + QString(e.what()));
        emit trainingFinished(-1);
    }
}

// 停止模型训练
void yoloTrainer::stopTraining() {
    if(c.running()){
        terminateProcessTree(c.id());
        emit logMessage("Training process terminated.");
        stopFlag.store(true);
        emit trainingFinished(-1);
    }
}


void yoloTrainer::terminateProcessTree(boost::process::pid_t pid){
#ifdef _WIN32
    std::string command = "taskkill /PID " + std::to_string(pid) + " /T /F";
    system(command.c_str());
#else
    std::string command = "kill -9 $(pstree -p " + std::to_string(pid) + " | grep -o '[0-9]*')";
    system(command.c_str());
#endif
}


