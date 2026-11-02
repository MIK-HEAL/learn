#include "window.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

void framebuffer_size_callback(GLFWwindow* window, int windowWidth, int windowHeight){
    (void)window;

    if (windowWidth <= 0 || windowHeight <= 0) {
        return;
    }

    glViewport(0, 0, windowWidth, windowHeight);
}


Window::Window() : window(nullptr) {
    setupWindow();
}

Window::~Window(){
    if (window != nullptr) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    glfwTerminate();
}

bool Window::shouldClose() const {
    return window == nullptr || glfwWindowShouldClose(window);
}

void Window::processInput(){
    if (window != nullptr && glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

void Window::swapBuffers(){
    if (window != nullptr) {
        glfwSwapBuffers(window);
    }
}

void Window::pollEvents(){
    glfwPollEvents();
}

GLFWwindow* Window::nativeHandle() const {
    return window;
}

void Window::setupWindow() {
    if (!glfwInit()) return;

    // 升級到 4.6 Core Profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(800, 600, "Hachimi", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create 4.6 context, trying 4.5..." << std::endl;
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        window = glfwCreateWindow(800, 600, "Hachimi", nullptr, nullptr);
    }

    if (window == nullptr) {
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwDestroyWindow(window);
        window = nullptr;
        glfwTerminate();
        return;
    }

    framebuffer_size_callback(window, 800, 600);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
}