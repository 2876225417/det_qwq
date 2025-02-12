#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>


class Configuration : public QWidget
{
    Q_OBJECT
public:
    explicit Configuration(QWidget *parent = nullptr);

signals:

private:
    QLabel* ConfigurationLabel;

};

#endif // CONFIGURATION_H
