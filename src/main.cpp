#include "window.hpp"
#include "ground.hpp"

#include <GL/glew.h>

int main()
{
    Window window;
    Ground ground;

    while (!window.shouldClose())
    {
        window.processInput();
        glClearColor(0.08f, 0.08f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ground.draw();
        window.swapBuffers();
        window.pollEvents();
    }

    return 0;
}
