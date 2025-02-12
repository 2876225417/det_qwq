#ifndef GPUINFO_H
#define GPUINFO_H


#include <QWidget>
#include <QVBoxLayout>
#include <QLabel>
#include <QGroupBox>


#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <nvml.h>

#include "utils/gpu_info/gpu_performance_monitor.h"


class GPU_INFO_PANEL: public QWidget {
    Q_OBJECT
public:
    explicit GPU_INFO_PANEL(QWidget* parent = nullptr);

protected:


private:


};


// singleton mode
class gpuinfo
{
public:
    std::vector<std::string> getGPUNames();
    std::vector<unsigned int> getGPUTemperatures();
    std::vector<size_t> getGPUMemoryUsages();

private:
    void initializeNvml();
    void shutdownNvml();
    bool nvmlInitialized;

public:
    static gpuinfo& gpuInfo();

private:
    gpuinfo();
    gpuinfo(const gpuinfo&) = delete;
    gpuinfo& operator=(const gpuinfo&) = delete;

};




#endif // GPUINFO_H
