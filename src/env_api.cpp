#include "ragdoll.hpp"

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

struct Environment {
    b2WorldId world{};
    b2BodyId ground{};
    std::unique_ptr<Ragdoll> ragdoll;
    std::mt19937 rng{0u};
    int stepCount{};
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

void resetEnvironment(Environment& environment) {
    environment.ragdoll.reset();
    environment.ragdoll = std::make_unique<Ragdoll>(environment.world);
    environment.ragdoll->randomize(environment.rng, kResetLinearNoise,
                                   kResetAngularNoise);
    environment.stepCount = 0;
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
