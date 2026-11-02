#include "window.hpp"
#include "ground.hpp"
#include "body.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <box2d/box2d.h>

int main()
{
    Window window;
    b2WorldDef worldDefinition = b2DefaultWorldDef();
    worldDefinition.gravity = {0.0f, -9.8f};
    const b2WorldId world = b2CreateWorld(&worldDefinition);
    {
        Ground ground(world);
        Body body(world);
        double previousTime = glfwGetTime();

        while (!window.shouldClose())
        {
            const double currentTime = glfwGetTime();
            const float deltaTime =
                static_cast<float>(currentTime - previousTime);
            previousTime = currentTime;

            window.processInput();
            b2World_Step(world, deltaTime, 4);
            glClearColor(0.08f, 0.08f, 0.08f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ground.draw();
            body.draw();
            window.swapBuffers();
            window.pollEvents();
        }
    }

    b2DestroyWorld(world);
    return 0;
}
