#include "serial_port_configuration_board.h"
#include "utils/gpu_info/gpu_performance_monitor.h"

serial_port_configuration_board
    ::serial_port_configuration_board(QWidget* parent)
    : QWidget{parent}

{
    QHBoxLayout* layout = new QHBoxLayout();
    test_info = new QLabel("serial_port_configuration_board", this);
    //gpu_performance_monitor_panel* panel = new gpu_performance_monitor_panel();
    //layout->addWidget(panel);
    layout->addWidget(test_info);
    setLayout(layout);
}
