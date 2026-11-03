#include "window.hpp"
#include "console.hpp"
#include "ground.hpp"
#include "body.hpp"
#include "ragdoll.hpp"
#include "policy.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <box2d/box2d.h>

#include <glm/vec2.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {
constexpr float kTimeStep = 1.0f / 60.0f;
constexpr int kSubStepCount = 4;
constexpr float kPolicyPollInterval = 0.5f; // 热重载轮询间隔（秒）
constexpr float kRespawnDelay = 1.5f;       // 摔倒后多久自动复位（秒）

// 顶点着色器把世界坐标映射为 px/400、-py/300 的 NDC（px = world * 50），
// 因此 NDC = ±1 对应世界坐标 ±8 米 / ±6 米，用于把鼠标位置还原成世界坐标。
constexpr float kWorldHalfWidth = 8.0f;
constexpr float kWorldHalfHeight = 6.0f;
// 单次冲量让被点部件产生的速度变化（m/s）。
// 取 4.0 是因为冲量按"被点部件的质量"换算：点躯干时整体约 1.6 m/s（明显晃动），
// 点手臂/小腿时该肢体会被甩开——1.5 太小，点轻部件几乎看不出反应。
// 按住左键会按 kClickRepeatInterval 连续施加，所以「点一下」和「按住猛推」都能做到。
constexpr float kClickSpeedChange = 4.0f;
constexpr float kClickRepeatInterval = 0.1f;

int runSelftest(const char* path) {
    Policy policy;
    if (!policy.load(path)) {
        console::eprint("策略加载失败: %s\n", path);
        return 1;
    }
    std::vector<float> observation(
        static_cast<std::size_t>(policy.observationCount()), 0.0f);
    std::vector<float> actions(
        static_cast<std::size_t>(policy.actionCount()), 0.0f);
    policy.forward(observation.data(), actions.data());
    console::print("策略校验通过: %s\n", path);
    console::print("obs=%d act=%d\n", policy.observationCount(), policy.actionCount());
    for (int index = 0; index < policy.actionCount(); ++index) {
        console::print("%.6f\n", actions[static_cast<std::size_t>(index)]);
    }
    return 0;
}

/// policyPath 为空时只跑物理；watch 为真时边训练边看（热重载 + 摔倒复位）。
int runViewer(const std::string& policyPath, bool watch) {
    Window window;

    Policy policy;
    bool policyLoaded = false;
    std::filesystem::file_time_type policyStamp{};
    if (!policyPath.empty()) {
        policyLoaded = policy.load(policyPath);
        if (!policyLoaded) {
            console::eprint("策略加载失败: %s\n", policyPath.c_str());
        } else {
            std::error_code error;
            policyStamp = std::filesystem::last_write_time(policyPath, error);
        }
    }
    const bool showPolicy = !policyPath.empty();
    if (showPolicy) {
        console::print("正在回放策略: %s%s\n", policyPath.c_str(),
                       watch ? "（监视文件更新，摔倒后自动复位）" : "");
    }
    console::print("操作提示: 鼠标左键点击角色的任意部位即可施加一次撞击\n");

    b2WorldDef worldDefinition = b2DefaultWorldDef();
    worldDefinition.gravity = {0.0f, -9.8f};
    worldDefinition.enableSleep = false;
    const b2WorldId world = b2CreateWorld(&worldDefinition);
    {
        Ground ground(world);
        auto ragdoll = std::make_unique<Ragdoll>(world);
        BodyRenderer renderer;

        float actions[Ragdoll::kActuatorCount]{};
        float observation[Ragdoll::kObsSize]{};
        float accumulator = 0.0f;
        float pollTimer = 0.0f;
        float fallenTimer = 0.0f;
        float clickCooldown = 0.0f;
        float hitMessageCooldown = 0.0f;
        double previousTime = glfwGetTime();

        while (!window.shouldClose())
        {
            const double currentTime = glfwGetTime();
            const auto frameDelta = static_cast<float>(currentTime - previousTime);
            accumulator += frameDelta;
            previousTime = currentTime;
            // 防止窗口卡顿（拖动/最小化）后一次性补太多物理步
            accumulator = std::min(accumulator, 0.25f);

            // 训练脚本每次刷新最佳成绩都会覆盖 policy.bin，这里热重载它
            pollTimer += frameDelta;
            if (watch && pollTimer >= kPolicyPollInterval) {
                pollTimer = 0.0f;
                std::error_code error;
                const auto stamp = std::filesystem::last_write_time(policyPath, error);
                if (!error && (!policyLoaded || stamp != policyStamp)) {
                    // 文件可能正被写入，失败就保留旧策略，下个周期再试
                    if (policy.load(policyPath)) {
                        policyStamp = stamp;
                        policyLoaded = true;
                        console::print("已重载策略: %s\n", policyPath.c_str());
                    }
                }
            }

            window.processInput();

            // 鼠标撞击：按住左键时对点中的部件持续施加冲量（点一下 = 推一下）
            clickCooldown = std::max(0.0f, clickCooldown - frameDelta);
            hitMessageCooldown = std::max(0.0f, hitMessageCooldown - frameDelta);
            if (GLFWwindow* handle = window.nativeHandle()) {
                const bool pressing =
                    glfwGetMouseButton(handle, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
                if (pressing && clickCooldown <= 0.0f) {
                    clickCooldown = kClickRepeatInterval;

                    double cursorX = 0.0;
                    double cursorY = 0.0;
                    glfwGetCursorPos(handle, &cursorX, &cursorY);
                    int windowWidth = 0;
                    int windowHeight = 0;
                    glfwGetWindowSize(handle, &windowWidth, &windowHeight);
                    if (windowWidth > 0 && windowHeight > 0) {
                        const glm::vec2 worldPoint{
                            (2.0f * static_cast<float>(cursorX) /
                                 static_cast<float>(windowWidth) -
                             1.0f) *
                                kWorldHalfWidth,
                            (1.0f - 2.0f * static_cast<float>(cursorY) /
                                        static_cast<float>(windowHeight)) *
                                kWorldHalfHeight,
                        };
                        const int partIndex = ragdoll->findPartAt(worldPoint);
                        if (partIndex >= 0) {
                            const glm::vec2 target = ragdoll->partPosition(partIndex);
                            // 朝身体内部推：方向 = 点击点 -> 部件质心
                            ragdoll->applyImpulse(
                                partIndex, worldPoint,
                                {target.x - worldPoint.x, target.y - worldPoint.y},
                                kClickSpeedChange);
                            if (hitMessageCooldown <= 0.0f) {
                                hitMessageCooldown = 0.5f;
                                console::print("撞击部件 #%d 质量 %.2f kg\n", partIndex,
                                               ragdoll->partMass(partIndex));
                            }
                        }
                    }
                }
            }

            while (accumulator >= kTimeStep)
            {
                if (policy.valid()) {
                    ragdoll->getObservation(observation);
                    policy.forward(observation, actions);
                }
                ragdoll->setActions(actions);
                b2World_Step(world, kTimeStep, kSubStepCount);
                accumulator -= kTimeStep;
            }

            // 回放策略时摔倒就自动复位，方便持续观察当前水平
            if (showPolicy) {
                if (ragdoll->fallen()) {
                    fallenTimer += frameDelta;
                    if (fallenTimer >= kRespawnDelay) {
                        ragdoll = std::make_unique<Ragdoll>(world);
                        fallenTimer = 0.0f;
                    }
                } else {
                    fallenTimer = 0.0f;
                }
            }

            glClearColor(0.08f, 0.08f, 0.08f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ground.draw();
            renderer.draw(*ragdoll);
            window.swapBuffers();
            window.pollEvents();
        }
    }

    b2DestroyWorld(world);
    return 0;
}
} // namespace

/// 用法：
///   window                              只跑物理（布娃娃会自然倒下）
///   window py/policy.bin                回放训练好的策略，摔倒后自动复位
///   window --watch py/policy.bin        边训练边看：热重载 policy.bin
///   window --selftest py/policy.bin     不建窗口，只验证策略文件
int main(int argc, char** argv)
{
    // 控制台默认是 GBK 代码页，先把 stdout/stderr 调成能正确显示中文的模式
    console::init();

    std::string policyPath;
    bool watch = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--selftest") {
            return index + 1 < argc ? runSelftest(argv[index + 1]) : 1;
        }
        if (argument == "--watch") {
            watch = true;
            continue;
        }
        policyPath = argument;
    }

    return runViewer(policyPath, watch);
}
