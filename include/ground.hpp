#pragma once

#include <box2d/box2d.h>
#include <glm/vec2.hpp>

class Ground {
public:
    explicit Ground(b2WorldId world);
    ~Ground();

    void draw() const;

private:
    b2WorldId world_{};
    b2BodyId body_{};
};