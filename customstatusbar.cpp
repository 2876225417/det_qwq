#include "customstatusbar.h"

CustomStatusBar::CustomStatusBar(QWidget* parent)
    : QStatusBar(parent)
{
    statusLabel = new QLabel("status: ", this);
    addWidget(statusLabel);
}

void CustomStatusBar::showStatusMessage(const QString& message, int timeout){
    showMessage(message, timeout);
}

void CustomStatusBar::addCustomWidget(QWidget* widget){
    addPermanentWidget(widget);
}
