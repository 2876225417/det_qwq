#include "openglboard.h"
#include <QOpenGLShader>
#include <QOpenGLShader>
#include <QMatrix4x4>
#include <QTimerEvent>

OpenGLBoard::OpenGLBoard(QWidget* parent):
    QOpenGLWidget(parent)
    , m_shaderProgram(nullptr)
    , m_progress(0.f){
    startTimer(50);
}

OpenGLBoard::~OpenGLBoard(){
    delete m_shaderProgram;
}

void OpenGLBoard::initializeGL(){
    initializeOpenGLFunctions();

    glClearColor(
        0.2f
        , 0.3f
        , 0.3f
        , 1.f
        );

    m_shaderProgram = new QOpenGLShaderProgram(this);
    m_shaderProgram->addShaderFromSourceFile(
        QOpenGLShader::Vertex,
        ":gl/gl/vertex_shader.vsh"
        );

    m_shaderProgram->addShaderFromSourceFile(
        QOpenGLShader::Fragment,
        ":gl/gl/fragment_shader.fsh"
        );
    m_shaderProgram->link();

    GLfloat vertices[] = {
        // 背景矩形（进度条的框架）
        -0.8f,  0.1f, 0.0f,  // 左上角
        -0.8f, -0.1f, 0.0f,  // 左下角
        0.8f, -0.1f, 0.0f,  // 右下角
        0.8f,  0.1f, 0.0f,  // 右上角

        // 填充矩形（进度条的填充部分）
        -0.8f,  0.1f, 0.0f,  // 左上角
        -0.8f, -0.1f, 0.0f,  // 左下角
        0.0f, -0.1f, 0.0f,  // 右下角
        0.0f,  0.1f, 0.0f   // 右上角
    };

    m_vao.create();
    m_vao.bind();

    m_vbo.create();
    m_vbo.bind();
    m_vbo.allocate(vertices, sizeof(vertices));

    m_shaderProgram->enableAttributeArray(0);
    m_shaderProgram->setAttributeArray(
        0
        , GL_FLOAT
        , reinterpret_cast<void*>(0)
        , 3
        );

    m_vbo.release();
    m_vao.release();
}

void OpenGLBoard::resizeGL(int w, int h){
    glViewport(0, 0, w, h);
}

void OpenGLBoard::paintGL(){
    glClear(GL_COLOR_BUFFER_BIT);

    m_shaderProgram->bind();

    // 更新进度条的填充矩形
    GLfloat vertices[] = {
        // 背景矩形（进度条的框架）
        -0.8f,  0.1f, 0.0f,
        -0.8f, -0.1f, 0.0f,
        0.8f, -0.1f, 0.0f,
        0.8f,  0.1f, 0.0f,

        // 填充矩形（根据进度更新）
        -0.8f,  0.1f, 0.0f,
        -0.8f, -0.1f, 0.0f,
        -0.8f + m_progress * 1.6f, -0.1f, 0.0f,  // 填充的右边缘（基于进度）
        -0.8f + m_progress * 1.6f,  0.1f, 0.0f   // 填充的右上角
    };
    m_vao.bind();
    m_vbo.bind();
    m_vbo.allocate(vertices, sizeof(vertices));

    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glDrawArrays(GL_TRIANGLE_FAN, 4, 4);

    m_vao.release();
    m_shaderProgram->release();
}

void OpenGLBoard::timerEvent(QTimerEvent* event){
    m_progress += 0.01f;
    if(m_progress >= 1.f){
        m_progress = 0.f;
    }
    update();
}





