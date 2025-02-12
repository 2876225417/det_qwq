#ifndef SERIAL_PORT_CONFIGURATION_BOARD_H
#define SERIAL_PORT_CONFIGURATION_BOARD_H

#include <QLabel>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>
#include "utils/gpu_info/gpu_performance_monitor.h"

class serial_port_configuration_board: public QWidget
{
    Q_OBJECT
public:
    explicit serial_port_configuration_board(QWidget *parent = nullptr);


private:
    QLabel* test_info;
};

#endif // SERIAL_PORT_CONFIGURATION_BOARD_H
