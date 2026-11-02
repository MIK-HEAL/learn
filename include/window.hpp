#pragma once

struct GLFWwindow;

class Window{
public:
    Window();
    ~Window();

    bool shouldClose() const;
    void processInput();
    void swapBuffers();
    void pollEvents();
    GLFWwindow* nativeHandle() const;

private:
    void setupWindow();

    GLFWwindow* window;
};