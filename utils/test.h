#ifndef TEST_H
#define TEST_H

#include <QDebug>

class test
{
public:
    test();

    void show() {
        qDebug() << "Hello from Qt!!";
    }
};

#endif // TEST_H
