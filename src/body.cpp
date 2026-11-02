#include "body.hpp"

#include "ragdoll.hpp"

#include <GL/glew.h>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>

#include <cstddef>
#include <iostream>

namespace {
constexpr float pixelsPerMeter = 50.0f;

glm::vec2 toScreen(b2Vec2 position) {
    return {
        position.x * pixelsPerMeter,
        -position.y * pixelsPerMeter
    };
}

GLuint createBodyShader() {
    constexpr const char* vertexSource = R"(
        #version 450 core
        layout (location = 0) in vec2 position;
        layout (location = 1) in vec2 localPosition;
        out vec2 localCoord;
        uniform vec2 halfExtent;
        void main() {
            gl_Position = vec4(position.x / 400.0,
                               -position.y / 300.0, 0.0, 1.0);
            localCoord = localPosition * halfExtent;
        }
    )";

    constexpr const char* fragmentSource = R"(
        #version 450 core
        in vec2 localCoord;
        uniform vec2 halfExtent;
        uniform float cornerRadius;
        uniform vec4 bodyColor;
        out vec4 color;
        void main() {
            vec2 q = abs(localCoord) - (halfExtent - vec2(cornerRadius));
            float distanceToEdge = length(max(q, vec2(0.0)))
                                 + min(max(q.x, q.y), 0.0)
                                 - cornerRadius;
            float alpha = 1.0 - smoothstep(-1.0, 1.0, distanceToEdge);
            if (alpha <= 0.01) discard;
            color = vec4(bodyColor.rgb, bodyColor.a * alpha);
        }
    )";

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, nullptr);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, nullptr);
    glCompileShader(fragmentShader);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    GLint linked{};
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_FALSE) {
        std::cerr << "Failed to link body shader\n";
        glDeleteProgram(program);
        return 0;
    }
    return program;
}
} // namespace

BodyRenderer::BodyRenderer() : program_(createBodyShader()) {
    if (program_ == 0) {
        return;
    }

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * 8, nullptr,
                 GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2) * 2,
                          nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2) * 2,
                          reinterpret_cast<void*>(sizeof(glm::vec2)));
    glEnableVertexAttribArray(1);
}

BodyRenderer::~BodyRenderer() {
    if (vbo_ != 0) {
        glDeleteBuffers(1, &vbo_);
    }
    if (vao_ != 0) {
        glDeleteVertexArrays(1, &vao_);
    }
    if (program_ != 0) {
        glDeleteProgram(program_);
    }
}

void BodyRenderer::draw(const Ragdoll& ragdoll) const {
    if (program_ == 0 || vao_ == 0) {
        return;
    }

    struct Vertex {
        glm::vec2 position;
        glm::vec2 localPosition;
    };

    glUseProgram(program_);
    glUniform4f(glGetUniformLocation(program_, "bodyColor"),
                0.2f, 0.75f, 1.0f, 1.0f);
    glUniform1f(glGetUniformLocation(program_, "cornerRadius"),
                0.12f * pixelsPerMeter);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);

    for (const Ragdoll::Part& part : ragdoll.parts()) {
        const b2Vec2 position = b2Body_GetPosition(part.body);
        const float angle = b2Rot_GetAngle(b2Body_GetRotation(part.body));
        const float cosine = glm::cos(angle);
        const float sine = glm::sin(angle);
        const glm::vec2 localCorners[] = {
            {-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 1.0f}, {-1.0f, 1.0f}
        };
        Vertex vertices[4]{};
        for (std::size_t index = 0; index < 4; ++index) {
            const glm::vec2 local = localCorners[index] * part.halfSize;
            const glm::vec2 rotated{
                cosine * local.x - sine * local.y,
                sine * local.x + cosine * local.y
            };
            vertices[index] = {
                toScreen({position.x + rotated.x, position.y + rotated.y}),
                localCorners[index]
            };
        }

        const glm::vec2 halfExtent = part.halfSize * pixelsPerMeter;
        glUniform2f(glGetUniformLocation(program_, "halfExtent"),
                    halfExtent.x, halfExtent.y);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
}
