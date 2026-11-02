#pragma once

#include <glm/vec2.hpp>

struct Ground {
    float left{-1000.0f};
    float right{1000.0f};
    float top{200.0f};
    float thickness{100.0f};

    bool collide(glm::vec2& position,
                 glm::vec2 halfSize,
                 glm::vec2& velocity) const;

    void draw();
};