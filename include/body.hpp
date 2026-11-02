#pragma once

#include <box2d/box2d.h>
#include <glm/vec2.hpp>
#include <vector>

class Body {
public:
    explicit Body(b2WorldId world);
    ~Body();

    void draw() const;

private:
    struct Part {
        b2BodyId body{};
        glm::vec2 halfSize{};
    };

    b2WorldId world_{};
    std::vector<Part> parts_;
    std::vector<b2JointId> joints_;

    b2BodyId createPart(glm::vec2 position, glm::vec2 halfSize);
    void connect(b2BodyId first, b2BodyId second, glm::vec2 anchor);
};