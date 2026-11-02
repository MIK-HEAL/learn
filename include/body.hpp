#pragma once

#include <glm/vec2.hpp>

struct Bone {
    glm::vec2 localPosition{};
    float localRotation{0.0f};
    float length{1.0f};

    Bone* parent{nullptr};
};

struct Joint{
    Bone* A, B;

};

class Body {
public:
    Body();

    void updateArmIK(glm::vec2 target);

    Bone torso;
    Bone leftUpperArm;
    Bone leftForeArm;
    Bone rightUpperArm;
    Bone rightForeArm;
    Bone leftThigh;
    Bone leftCalf;
    Bone rightThigh;
    Bone rightCalf;
    Bone head;

    void update(float deltaTime, const Ground& ground);
    glm::vec2 position{0.0f, -100.0f};
    glm::vec2 velocity{};
    glm::vec2 halfSize{25.0f, 50.0f};

    bool onGround{false};

    glm::vec2 handPosition{};

private:
    glm::vec2 getWorldPosition(const Bone& bone) const;
};