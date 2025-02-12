#include "configuration.h"

Configuration::Configuration(QWidget *parent)
    : QWidget{parent}
{
    ConfigurationLabel = new QLabel("configuration", this);

    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(ConfigurationLabel);
    setLayout(layout);




}
