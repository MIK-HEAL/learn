#include "window.hpp"
#include "ground.hpp"
#include "body.hpp"
#include "ragdoll.hpp"
#include "policy.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <box2d/box2d.h>

#include <algorithm>
#include <cstdio>

namespace {
constexpr float kTimeStep = 1.0f / 60.0f;
constexpr int kSubStepCount = 4;
} // namespace

/// 用法：
///   window                       只跑物理（布娃娃会自然倒下）
///   window py/policy.bin         加载训练好的策略，让角色站起来
int main(int argc, char** argv)
{
    Window window;

    Policy policy;
    if (argc > 1 && !policy.load(argv[1])) {
        std::fprintf(stderr, "策略加载失败: %s\n", argv[1]);
    }

    b2WorldDef worldDefinition = b2DefaultWorldDef();
    worldDefinition.gravity = {0.0f, -9.8f};
    worldDefinition.enableSleep = false;
    const b2WorldId world = b2CreateWorld(&worldDefinition);
    {
        Ground ground(world);
        Ragdoll ragdoll(world);
        BodyRenderer renderer;

        float actions[Ragdoll::kActuatorCount]{};
        float observation[Ragdoll::kObsSize]{};
        float accumulator = 0.0f;
        double previousTime = glfwGetTime();

        while (!window.shouldClose())
        {
            const double currentTime = glfwGetTime();
            accumulator += static_cast<float>(currentTime - previousTime);
            previousTime = currentTime;
            // 防止窗口卡顿（拖动/最小化）后一次性补太多物理步
            accumulator = std::min(accumulator, 0.25f);

            window.processInput();
            while (accumulator >= kTimeStep)
            {
                if (policy.valid()) {
                    ragdoll.getObservation(observation);
                    policy.forward(observation, actions);
                }
                ragdoll.setActions(actions);
                b2World_Step(world, kTimeStep, kSubStepCount);
                accumulator -= kTimeStep;
            }

            glClearColor(0.08f, 0.08f, 0.08f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ground.draw();
            renderer.draw(ragdoll);
            window.swapBuffers();
            window.pollEvents();
        }
    }

    b2DestroyWorld(world);
    return 0;
}
