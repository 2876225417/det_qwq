#ifndef OPENGLBOARD_H
#define OPENGLBOARD_H

#include <QOpenglFunctions>
#include <QOpenGLWidget>
#include <QOpenGLShaderProgram>
#include <QTimer>
#include <QOpenGLBuffer>
#include <QOpenGLTexture>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLFunctions_4_5_Core>

class OpenGLBoard: public QOpenGLWidget,  protected QOpenGLFunctions_4_5_Core
{
    Q_OBJECT
public:
    explicit OpenGLBoard(QWidget* parent = nullptr);
    ~OpenGLBoard();

protected:
    void initializeGL() override;

    void resizeGL(int w, int h) override;

    void paintGL() override;

    void timerEvent(QTimerEvent* event) override;

private:
    QOpenGLShaderProgram* m_shaderProgram;
    QOpenGLBuffer m_vbo;
    QOpenGLVertexArrayObject m_vao;
    float m_progress;
};

#endif // OPENGLBOARD_H
