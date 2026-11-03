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
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {
constexpr float kTimeStep = 1.0f / 60.0f;
constexpr int kSubStepCount = 4;
constexpr float kPolicyPollInterval = 0.5f; // 热重载轮询间隔（秒）
constexpr float kRespawnDelay = 1.5f;       // 摔倒后多久自动复位（秒）

int runSelftest(const char* path) {
    Policy policy;
    if (!policy.load(path)) {
        return 1;
    }
    std::vector<float> observation(
        static_cast<std::size_t>(policy.observationCount()), 0.0f);
    std::vector<float> actions(
        static_cast<std::size_t>(policy.actionCount()), 0.0f);
    policy.forward(observation.data(), actions.data());
    std::printf("obs=%d act=%d\n", policy.observationCount(), policy.actionCount());
    for (int index = 0; index < policy.actionCount(); ++index) {
        std::printf("%.6f\n", actions[static_cast<std::size_t>(index)]);
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
            std::fprintf(stderr, "策略加载失败: %s\n", policyPath.c_str());
        } else {
            std::error_code error;
            policyStamp = std::filesystem::last_write_time(policyPath, error);
        }
    }
    const bool showPolicy = !policyPath.empty();
    if (showPolicy) {
        std::printf("正在回放策略: %s%s\n", policyPath.c_str(),
                    watch ? "（监视文件更新，摔倒后自动复位）" : "");
    }

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
                        std::printf("已重载策略: %s\n", policyPath.c_str());
                    }
                }
            }

            window.processInput();
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
