#pragma once

/// 只负责把 Ragdoll 画出来的渲染器（OpenGL 资源由它在构造时创建）。
class Ragdoll;

class BodyRenderer {
public:
    BodyRenderer();
    ~BodyRenderer();

    BodyRenderer(const BodyRenderer&) = delete;
    BodyRenderer& operator=(const BodyRenderer&) = delete;

    void draw(const Ragdoll& ragdoll) const;

private:
    unsigned int program_{};
    unsigned int vao_{};
    unsigned int vbo_{};
};
