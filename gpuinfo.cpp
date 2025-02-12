#include "gpuinfo.h"

#include "utils/gpu_info/gpu_performance_monitor.h"
#include <QDebug>


GPU_INFO_PANEL::GPU_INFO_PANEL(QWidget* parent)
    : QWidget(parent) {

    QVBoxLayout* gpu__info__panel = new QVBoxLayout();

    QVBoxLayout* gpu_info = new QVBoxLayout();

    QLabel* test = new QLabel("NVIDIA CUDA Test");


    // gpu_performance_monitor_panel* test_ = new gpu_performance_monitor_panel();
    gpu_info->addWidget(test);
    // gpu_info->addWidget(test_);

    QGroupBox* gpu_info_wrapper = new QGroupBox("GPUInfo");
    gpu_info_wrapper->setLayout(gpu_info);

    std::vector<std::string> gpuNames = gpuinfo::gpuInfo().getGPUNames();
    qDebug() << gpuNames;

    qDebug() << gpuinfo::gpuInfo().getGPUTemperatures();

    gpu__info__panel->addWidget(gpu_info_wrapper);
    setLayout(gpu__info__panel);


}


gpuinfo::gpuinfo()
{
    initializeNvml();
}

gpuinfo& gpuinfo::gpuInfo(){
    static gpuinfo gpuInfo;
    return gpuInfo;
}

void gpuinfo::initializeNvml(){
    if(nvmlInitialized) return;

    nvmlReturn_t result = nvmlInit();
    if(result != NVML_SUCCESS){
        qDebug() << "Failed to initialize NVML: " << nvmlErrorString(result);
        return;
    }
    nvmlInitialized = true;
}

void gpuinfo::shutdownNvml(){
    if(nvmlInitialized){
        nvmlShutdown();
        nvmlInitialized = false;
    }
}

std::vector<std::string> gpuinfo::getGPUNames(){
    std::vector<std::string> names;
    if(!nvmlInitialized){
        qDebug() << "NVML not initialized";
        return names;
    }

    unsigned int deviceCount;
    nvmlDevice_t device;
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    nvmlReturn_t result = nvmlDeviceGetCount(&deviceCount);
    if(result != NVML_SUCCESS){
        qDebug() << "Failed to get device count: " << nvmlErrorString(result);
        return names;
    }

    for(unsigned int i = 0; i < deviceCount; i++){
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if(result != NVML_SUCCESS){
            qDebug() << "Failed to get handle for device " << i << ": " << nvmlErrorString(result);
            continue;
        }

        result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        if(result != NVML_SUCCESS){
            qDebug() << "Failed to get name for device " << i << ": "  << nvmlErrorString(result);
            continue;
        }

        names.push_back(std::string(name));
    }
    return names;
}


std::vector<unsigned int> gpuinfo::getGPUTemperatures(){
    std::vector<unsigned int> temperatures;
    if(!nvmlInitialized){
        qDebug() << "NVML not initialized";
        return temperatures;
    }

    unsigned int deviceCount;
    nvmlDevice_t device;

    nvmlReturn_t result = nvmlDeviceGetCount(&deviceCount);
    if(result != NVML_SUCCESS){
        qDebug() << "Failed to get device count: " << nvmlErrorString(result);
        return temperatures;
    }

    for(unsigned int i = 0; i < deviceCount; ++i){
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if(result != NVML_SUCCESS){
            qDebug() << "Failed to get handle for device " << i << ": " << nvmlErrorString(result);
            continue;
        }

        unsigned int temp;
        result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
        if(result != NVML_SUCCESS){
            qDebug() << "Failed to get temperature for device " << i << ": " << nvmlErrorString(result);
            continue;
        }

        temperatures.push_back(temp);
    }
    return temperatures;
}

std::vector<size_t> gpuinfo::getGPUMemoryUsages(){
    std::vector<size_t> memoryUsages;
    if(!nvmlInitialized){
        qDebug() << "NVML not initialized";
        return memoryUsages;
    }

    unsigned int deviceCount;
    nvmlDevice_t device;
    nvmlMemory_t memoryInfo;

    nvmlReturn_t result = nvmlDeviceGetCount(&deviceCount);
    if(result != NVML_SUCCESS){
        qDebug() << "Failed to get device count: " << nvmlErrorString(result);
        return memoryUsages;
    }

    for(unsigned int i = 0; i < deviceCount; ++i){
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if(result != NVML_SUCCESS){
            qDebug() << "Failed to get memory info for device " << i << ": " << nvmlErrorString(result);
            continue;
        }

        result = nvmlDeviceGetMemoryInfo(device, &memoryInfo);
        if(result != NVML_SUCCESS){
            qDebug() << "Failed to get memory info for device " << i << ":" << nvmlErrorString(result);
            continue;
        }
        memoryUsages.push_back(memoryInfo.used);
    }
    return memoryUsages;
}









