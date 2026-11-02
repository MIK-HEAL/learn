#include "ragdoll.hpp"

#include <algorithm>
#include <cmath>

namespace {
constexpr float kPi = 3.14159265358979f;

struct PartSpec {
    glm::vec2 position{};
    glm::vec2 halfSize{};
};

struct JointSpec {
    int parentIndex{};
    int childIndex{};
    glm::vec2 anchor{};
    bool actuated{false};
    float lower{};
    float upper{};
};

constexpr float kJointLimit = 2.4f;
constexpr float kNeckLimit = 0.5f;

// 与原先 Body 中的布局完全一致，只是把数据抽成表，方便训练环境复用。
const PartSpec kPartSpecs[Ragdoll::kPartCount] = {
    {{0.0f, 2.6f}, {0.7f, 1.0f}},      // 0 躯干
    {{0.0f, 4.0f}, {0.45f, 0.45f}},    // 1 头
    {{-1.0f, 2.9f}, {0.65f, 0.18f}},   // 2 左上臂
    {{-2.25f, 2.9f}, {0.65f, 0.16f}},  // 3 左小臂
    {{1.0f, 2.9f}, {0.65f, 0.18f}},    // 4 右上臂
    {{2.25f, 2.9f}, {0.65f, 0.16f}},   // 5 右小臂
    {{-0.4f, 1.1f}, {0.22f, 0.8f}},    // 6 左大腿
    {{-0.4f, -0.5f}, {0.2f, 0.8f}},    // 7 左小腿
    {{0.4f, 1.1f}, {0.22f, 0.8f}},     // 8 右大腿
    {{0.4f, -0.5f}, {0.2f, 0.8f}},     // 9 右小腿
};

const JointSpec kJointSpecs[Ragdoll::kJointCount] = {
    {0, 1, {0.0f, 3.6f}, false, -kNeckLimit, kNeckLimit},    // 颈
    {0, 2, {-0.65f, 3.2f}, true, -kJointLimit, kJointLimit}, // 左肩
    {2, 3, {-1.65f, 2.9f}, true, -kJointLimit, kJointLimit}, // 左肘
    {0, 4, {0.65f, 3.2f}, true, -kJointLimit, kJointLimit},  // 右肩
    {4, 5, {1.65f, 2.9f}, true, -kJointLimit, kJointLimit},  // 右肘
    {0, 6, {-0.4f, 1.65f}, true, -kJointLimit, kJointLimit}, // 左髋
    {6, 7, {-0.4f, 0.3f}, true, -kJointLimit, kJointLimit},  // 左膝
    {0, 8, {0.4f, 1.65f}, true, -kJointLimit, kJointLimit},  // 右髋
    {8, 9, {0.4f, 0.3f}, true, -kJointLimit, kJointLimit},   // 右膝
};
} // namespace

Ragdoll::Ragdoll(b2WorldId world, float verticalOffset) : world_(world) {
    parts_.reserve(kPartCount);
    for (const PartSpec& spec : kPartSpecs) {
        parts_.push_back(createPart(
            {spec.position.x, spec.position.y + verticalOffset}, spec.halfSize));
    }

    joints_.reserve(kJointCount);
    for (const JointSpec& spec : kJointSpecs) {
        const glm::vec2 anchor{spec.anchor.x, spec.anchor.y + verticalOffset};
        const b2JointId joint =
            connect(parts_[spec.parentIndex].body, parts_[spec.childIndex].body, anchor);
        b2RevoluteJoint_EnableLimit(joint, true);
        b2RevoluteJoint_SetLimits(joint, spec.lower, spec.upper);
        b2RevoluteJoint_SetMaxMotorTorque(joint, kMotorTorque);
        b2RevoluteJoint_SetMotorSpeed(joint, 0.0f);
        b2RevoluteJoint_EnableMotor(joint, spec.actuated);
        joints_.push_back(joint);
    }

    // 执行器 index 对应关节 index + 1（跳过不可驱动的颈部）
    for (int index = 0; index < kActuatorCount; ++index) {
        actuatorJoints_[static_cast<std::size_t>(index)] =
            joints_[static_cast<std::size_t>(index + 1)];
    }
}

Ragdoll::~Ragdoll() {
    for (const b2JointId joint : joints_) {
        if (B2_IS_NON_NULL(joint)) {
            b2DestroyJoint(joint);
        }
    }
    for (const Part& part : parts_) {
        if (B2_IS_NON_NULL(part.body)) {
            b2DestroyBody(part.body);
        }
    }
}

Ragdoll::Part Ragdoll::createPart(glm::vec2 position, glm::vec2 halfSize) {
    b2BodyDef bodyDefinition = b2DefaultBodyDef();
    bodyDefinition.type = b2_dynamicBody;
    bodyDefinition.position = {position.x, position.y};
    bodyDefinition.linearDamping = 0.1f;
    bodyDefinition.angularDamping = 0.2f;
    const b2BodyId body = b2CreateBody(world_, &bodyDefinition);

    b2ShapeDef shapeDefinition = b2DefaultShapeDef();
    shapeDefinition.density = 1.0f;
    shapeDefinition.material.friction = 0.7f;
    const b2Polygon box = b2MakeBox(halfSize.x, halfSize.y);
    b2CreatePolygonShape(body, &shapeDefinition, &box);

    return {body, halfSize};
}

b2JointId Ragdoll::connect(b2BodyId first, b2BodyId second, glm::vec2 anchor) {
    b2RevoluteJointDef jointDefinition = b2DefaultRevoluteJointDef();
    jointDefinition.bodyIdA = first;
    jointDefinition.bodyIdB = second;
    jointDefinition.collideConnected = false;
    const b2Vec2 pivot{anchor.x, anchor.y};
    jointDefinition.localAnchorA = b2Body_GetLocalPoint(first, pivot);
    jointDefinition.localAnchorB = b2Body_GetLocalPoint(second, pivot);
    return b2CreateRevoluteJoint(world_, &jointDefinition);
}

void Ragdoll::setActions(const float* actions) {
    if (actions == nullptr) {
        return;
    }
    for (int index = 0; index < kActuatorCount; ++index) {
        const float value = std::clamp(actions[index], -1.0f, 1.0f);
        b2RevoluteJoint_SetMotorSpeed(
            actuatorJoints_[static_cast<std::size_t>(index)], value * kMotorSpeedScale);
    }
}

void Ragdoll::randomize(std::mt19937& rng, float linearScale, float angularScale) {
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    for (const Part& part : parts_) {
        b2Body_SetLinearVelocity(part.body, {uniform(rng) * linearScale,
                                             uniform(rng) * linearScale});
        b2Body_SetAngularVelocity(part.body, uniform(rng) * angularScale);
    }
}

void Ragdoll::getObservation(float* observation) const {
    if (observation == nullptr) {
        return;
    }

    const b2Vec2 basePositionValue = b2Body_GetPosition(parts_[0].body);
    const b2Vec2 baseUp = b2Body_GetWorldVector(parts_[0].body, {0.0f, 1.0f});
    const b2Vec2 baseVelocity = b2Body_GetLinearVelocity(parts_[0].body);

    int index = 0;
    observation[index++] = basePositionValue.x;
    observation[index++] = basePositionValue.y;
    observation[index++] = baseUp.x;
    observation[index++] = baseUp.y;
    observation[index++] = baseVelocity.x;
    observation[index++] = baseVelocity.y;
    observation[index++] = b2Body_GetAngularVelocity(parts_[0].body);

    for (const Part& part : parts_) {
        const b2Vec2 position = b2Body_GetPosition(part.body);
        const b2Vec2 up = b2Body_GetWorldVector(part.body, {0.0f, 1.0f});
        observation[index++] = position.x - basePositionValue.x;
        observation[index++] = position.y - basePositionValue.y;
        observation[index++] = up.x;
        observation[index++] = up.y;
    }

    for (const b2JointId joint : joints_) {
        observation[index++] = b2RevoluteJoint_GetAngle(joint) / kPi;
    }

    for (const Part& part : parts_) {
        observation[index++] = b2Body_GetAngularVelocity(part.body);
    }
}

float Ragdoll::reward(const float* actions) const {
    const b2Vec2 position = b2Body_GetPosition(parts_[0].body);
    const float upright = baseUprightness();
    // 躯干高度归一化到 [-1, 1]，站立时约为 1
    const float height =
        std::clamp((position.y - (kGroundTopY + 1.0f)) / 2.0f, -1.0f, 1.0f);

    float control = 0.0f;
    if (actions != nullptr) {
        for (int index = 0; index < kActuatorCount; ++index) {
            control += actions[index] * actions[index];
        }
        control /= static_cast<float>(kActuatorCount);
    }

    const float drift = std::abs(position.x);
    return 1.0f + 1.0f * upright + 0.5f * height - 0.03f * control - 0.02f * drift;
}

bool Ragdoll::fallen() const {
    return basePosition().y < kFallHeight || baseUprightness() < kFallUpY;
}

glm::vec2 Ragdoll::basePosition() const {
    const b2Vec2 position = b2Body_GetPosition(parts_[0].body);
    return {position.x, position.y};
}

float Ragdoll::baseUprightness() const {
    return b2Body_GetWorldVector(parts_[0].body, {0.0f, 1.0f}).y;
}
