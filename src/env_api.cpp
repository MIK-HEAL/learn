#include "ragdoll.hpp"

#include <glm/geometric.hpp>

#include <box2d/box2d.h>

#include <memory>
#include <random>

#if defined(_WIN32)
#define RL_ENV_EXPORT extern "C" __declspec(dllexport)
#else
#define RL_ENV_EXPORT extern "C" __attribute__((visibility("default")))
#endif

/// 供 Python（ctypes）调用的 C 接口。所有函数都不做异常抛出，
/// 因此 Python 侧只需要按顺序调用 env_create / env_reset / env_step 即可。
namespace {
constexpr float kTimeStep = 1.0f / 60.0f;
constexpr int kSubStepCount = 4;
constexpr int kMaxEpisodeSteps = 400; // 6.67 秒

constexpr float kResetLinearNoise = 0.35f;
constexpr float kResetAngularNoise = 1.0f;

// 随机扰动（抗扰动训练）：每隔 1.5~4 秒来一次，有两种形态
//   1) 瞬时冲量（像被球砸一下）
//   2) 持续推力段（像演示里按住左键一直推）
// 强度刻意覆盖演示端「单击 4.0 m/s」的量级，否则策略在演示里会被一推就倒。
constexpr int kPushIntervalMinSteps = 90;
constexpr int kPushIntervalMaxSteps = 240;
constexpr float kPushSpeedMin = 1.5f;
constexpr float kPushSpeedMax = 5.0f;

constexpr float kHoldPushChance = 0.35f;      // 有多大概率是「持续推」而不是瞬时撞击
constexpr float kHoldAccelerationMin = 6.0f;  // m/s^2
constexpr float kHoldAccelerationMax = 12.0f;
constexpr int kHoldStepsMin = 18;             // 0.3 秒
constexpr int kHoldStepsMax = 40;             // 0.67 秒

struct Environment {
    b2WorldId world{};
    b2BodyId ground{};
    std::unique_ptr<Ragdoll> ragdoll;
    std::mt19937 rng{0u};
    int stepCount{};
    int nextPushStep{0};
    // 进行中的持续推力段
    int holdPartIndex{-1};
    b2Vec2 holdDirection{1.0f, 0.0f};
    float holdAcceleration{0.0f};
    int holdStepsLeft{0};
};

b2WorldId createWorld() {
    b2WorldDef worldDefinition = b2DefaultWorldDef();
    worldDefinition.gravity = {0.0f, -9.8f};
    worldDefinition.enableSleep = false; // 训练时避免物理休眠导致状态冻结
    return b2CreateWorld(&worldDefinition);
}

b2BodyId createGround(b2WorldId world) {
    b2BodyDef bodyDefinition = b2DefaultBodyDef();
    bodyDefinition.position = {0.0f, Ragdoll::kGroundTopY - 1.0f};
    const b2BodyId body = b2CreateBody(world, &bodyDefinition);

    b2ShapeDef shapeDefinition = b2DefaultShapeDef();
    shapeDefinition.material.friction = 0.8f;
    const b2Polygon box = b2MakeBox(20.0f, 1.0f);
    b2CreatePolygonShape(body, &shapeDefinition, &box);
    return body;
}

int scheduleNextPush(std::mt19937& rng) {
    std::uniform_int_distribution<int> interval(kPushIntervalMinSteps, kPushIntervalMaxSteps);
    return interval(rng);
}

int pickRandomPart(std::mt19937& rng) {
    std::uniform_int_distribution<int> distribution(0, Ragdoll::kPartCount - 1);
    return distribution(rng);
}

glm::vec2 randomDirection(std::mt19937& rng) {
    std::uniform_real_distribution<float> horizontal(-1.0f, 1.0f);
    std::uniform_real_distribution<float> vertical(-0.35f, 0.35f);
    glm::vec2 direction{horizontal(rng), vertical(rng)};
    const float length = glm::length(direction);
    if (length < 1e-3f) {
        return {1.0f, 0.0f};
    }
    return direction / length;
}

void resetEnvironment(Environment& environment) {
    environment.ragdoll.reset();
    environment.ragdoll = std::make_unique<Ragdoll>(environment.world);
    environment.ragdoll->randomize(environment.rng, kResetLinearNoise,
                                   kResetAngularNoise);
    environment.stepCount = 0;
    environment.nextPushStep = scheduleNextPush(environment.rng);
    environment.holdStepsLeft = 0;
    environment.holdPartIndex = -1;
}
} // namespace

RL_ENV_EXPORT int env_obs_size() {
    return Ragdoll::kObsSize;
}

RL_ENV_EXPORT int env_act_size() {
    return Ragdoll::kActuatorCount;
}

RL_ENV_EXPORT int env_max_episode_steps() {
    return kMaxEpisodeSteps;
}

RL_ENV_EXPORT float env_time_step() {
    return kTimeStep;
}

RL_ENV_EXPORT void* env_create() {
    auto* environment = new Environment();
    environment->world = createWorld();
    environment->ground = createGround(environment->world);
    resetEnvironment(*environment);
    return environment;
}

RL_ENV_EXPORT void env_destroy(void* handle) {
    auto* environment = static_cast<Environment*>(handle);
    if (environment == nullptr) {
        return;
    }
    environment->ragdoll.reset();
    if (B2_IS_NON_NULL(environment->ground)) {
        b2DestroyBody(environment->ground);
    }
    b2DestroyWorld(environment->world);
    delete environment;
}

RL_ENV_EXPORT void env_set_seed(void* handle, unsigned int seed) {
    auto* environment = static_cast<Environment*>(handle);
    if (environment == nullptr) {
        return;
    }
    environment->rng.seed(seed);
}

/// 对指定部件施加一次冲量（speedChange 是期望的速度变化，单位 m/s）。
/// 演示里的鼠标撞击和训练里的随机扰动都是同一套冲量逻辑，
/// 这里额外开放给 Python，方便做抗撞评估。
RL_ENV_EXPORT void env_poke(void* handle, int partIndex, float directionX,
                            float directionY, float speedChange) {
    auto* environment = static_cast<Environment*>(handle);
    if (environment == nullptr || !environment->ragdoll) {
        return;
    }
    const glm::vec2 position = environment->ragdoll->partPosition(partIndex);
    environment->ragdoll->applyImpulse(partIndex, position,
                                       {directionX, directionY}, speedChange);
}

RL_ENV_EXPORT void env_reset(void* handle, float* observation) {
    auto* environment = static_cast<Environment*>(handle);
    if (environment == nullptr) {
        return;
    }
    resetEnvironment(*environment);
    if (observation != nullptr) {
        environment->ragdoll->getObservation(observation);
    }
}

RL_ENV_EXPORT void env_step(void* handle, const float* actions, float* observation,
                            float* reward, int* done) {
    auto* environment = static_cast<Environment*>(handle);
    if (environment == nullptr) {
        return;
    }

    static const float kZeroActions[Ragdoll::kActuatorCount]{};
    const float* appliedActions = actions != nullptr ? actions : kZeroActions;

    environment->ragdoll->setActions(appliedActions);

    // 进行中的「按住推」：每个物理步都施加一次持续力
    if (environment->holdStepsLeft > 0) {
        environment->ragdoll->applyForce(
            environment->holdPartIndex,
            {environment->holdDirection.x, environment->holdDirection.y},
            environment->holdAcceleration);
        environment->holdStepsLeft -= 1;
    }

    // 到点触发一次新的扰动
    if (environment->stepCount >= environment->nextPushStep) {
        std::uniform_real_distribution<float> chance(0.0f, 1.0f);
        if (chance(environment->rng) < kHoldPushChance) {
            std::uniform_real_distribution<float> acceleration(kHoldAccelerationMin,
                                                               kHoldAccelerationMax);
            std::uniform_int_distribution<int> holdSteps(kHoldStepsMin, kHoldStepsMax);
            const glm::vec2 direction = randomDirection(environment->rng);
            environment->holdPartIndex = pickRandomPart(environment->rng);
            environment->holdDirection = {direction.x, direction.y};
            environment->holdAcceleration = acceleration(environment->rng);
            environment->holdStepsLeft = holdSteps(environment->rng);
        } else {
            std::uniform_real_distribution<float> strength(kPushSpeedMin, kPushSpeedMax);
            environment->ragdoll->applyRandomPush(environment->rng, strength(environment->rng));
        }
        environment->nextPushStep =
            environment->stepCount + scheduleNextPush(environment->rng);
    }

    b2World_Step(environment->world, kTimeStep, kSubStepCount);
    environment->stepCount += 1;

    const float stepReward = environment->ragdoll->reward(appliedActions);
    const bool terminated = environment->ragdoll->fallen();
    const bool truncated = environment->stepCount >= kMaxEpisodeSteps;

    if (reward != nullptr) {
        *reward = stepReward;
    }
    if (done != nullptr) {
        *done = (terminated || truncated) ? 1 : 0;
    }
    if (observation != nullptr) {
        environment->ragdoll->getObservation(observation);
    }
}
