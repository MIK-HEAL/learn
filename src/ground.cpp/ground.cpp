#include "ground.hpp"

#include <GL/glew.h>
#include <iostream>

namespace {
GLuint createShaderProgram() {
    constexpr const char* vertexSource = R"(
        #version 460 core
        layout (location = 0) in vec2 position;

        void main() {
            gl_Position = vec4(position.x / 400.0,
                               -position.y / 300.0,
                               0.0,
                               1.0);
        }
    )";

    constexpr const char* fragmentSource = R"(
        #version 460 core
        out vec4 color;

        void main() {
            color = vec4(1.0, 1.0, 1.0, 1.0);
        }
    )";

    const GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, nullptr);
    glCompileShader(vertexShader);

    const GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, nullptr);
    glCompileShader(fragmentShader);

    const GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    GLint linked{};
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_FALSE) {
        std::cerr << "Failed to link ground shader\n";
        glDeleteProgram(program);
        return 0;
    }

    return program;
}
}

bool Ground::collide(glm::vec2& position,
                     glm::vec2 halfSize,
                     glm::vec2& velocity) const {
    const float bottom = position.y + halfSize.y;
    const bool insideHorizontalRange =
        position.x + halfSize.x > left &&
        position.x - halfSize.x < right;

    if (!insideHorizontalRange || bottom < top || velocity.y < 0.0f) {
        return false;
    }

    position.y = top - halfSize.y;
    velocity.y = 0.0f;
    return true;
}

void Ground::draw() {
    static const GLuint program = createShaderProgram();
    static GLuint vao{};
    static GLuint vbo{};

    if (program == 0) {
        return;
    }

    if (vao == 0) {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 8, nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                              2 * sizeof(float), nullptr);
        glEnableVertexAttribArray(0);
    }

    const float vertices[] = {
        left, top,
        right, top,
        right, top + thickness,
        left, top + thickness
    };

    glUseProgram(program);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
}