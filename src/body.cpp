#include "body.hpp"

#include <GL/glew.h>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
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
        uniform float pointSize;

        void main() {
            gl_Position = vec4(position.x / 400.0,
                               -position.y / 300.0,
                               0.0,
                               1.0);
            gl_PointSize = pointSize;
        }
    )";

    constexpr const char* fragmentSource = R"(
        #version 450 core
        uniform vec4 bodyColor;
        out vec4 color;

        void main() {
            color = bodyColor;
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
        std::cerr << "Failed to link body shader\n";
        glDeleteProgram(program);
        return 0;
    }

    return program;
}
}

Body::Body(b2WorldId world) : world_(world) {
    const b2BodyId torso = createPart({0.0f, 2.6f}, {0.7f, 1.0f});
    const b2BodyId head = createPart({0.0f, 4.0f}, {0.45f, 0.45f});
    const b2BodyId leftUpperArm = createPart({-1.0f, 2.9f}, {0.65f, 0.18f});
    const b2BodyId leftForeArm = createPart({-2.25f, 2.9f}, {0.65f, 0.16f});
    const b2BodyId rightUpperArm = createPart({1.0f, 2.9f}, {0.65f, 0.18f});
    const b2BodyId rightForeArm = createPart({2.25f, 2.9f}, {0.65f, 0.16f});
    const b2BodyId leftThigh = createPart({-0.4f, 1.1f}, {0.22f, 0.8f});
    const b2BodyId leftShin = createPart({-0.4f, -0.5f}, {0.2f, 0.8f});
    const b2BodyId rightThigh = createPart({0.4f, 1.1f}, {0.22f, 0.8f});
    const b2BodyId rightShin = createPart({0.4f, -0.5f}, {0.2f, 0.8f});

    connect(torso, head, {0.0f, 3.6f});
    connect(torso, leftUpperArm, {-0.65f, 3.2f});
    connect(leftUpperArm, leftForeArm, {-1.65f, 2.9f});
    connect(torso, rightUpperArm, {0.65f, 3.2f});
    connect(rightUpperArm, rightForeArm, {1.65f, 2.9f});
    connect(torso, leftThigh, {-0.4f, 1.65f});
    connect(leftThigh, leftShin, {-0.4f, 0.3f});
    connect(torso, rightThigh, {0.4f, 1.65f});
    connect(rightThigh, rightShin, {0.4f, 0.3f});
}

Body::~Body() {
    for (b2JointId joint : joints_) {
        b2DestroyJoint(joint);
    }
    for (const Part& part : parts_) {
        b2DestroyBody(part.body);
    }
}

b2BodyId Body::createPart(glm::vec2 position, glm::vec2 halfSize) {
    b2BodyDef bodyDef = b2DefaultBodyDef();
    bodyDef.type = b2_dynamicBody;
    bodyDef.position = {position.x, position.y};
    bodyDef.linearDamping = 0.1f;
    bodyDef.angularDamping = 0.2f;
    const b2BodyId body = b2CreateBody(world_, &bodyDef);

    b2ShapeDef shapeDef = b2DefaultShapeDef();
    shapeDef.density = 1.0f;
    shapeDef.material.friction = 0.7f;
    const b2Polygon box = b2MakeBox(halfSize.x, halfSize.y);
    b2CreatePolygonShape(body, &shapeDef, &box);

    parts_.push_back({body, halfSize});
    return body;
}

void Body::connect(b2BodyId first, b2BodyId second, glm::vec2 anchor) {
    b2RevoluteJointDef jointDef = b2DefaultRevoluteJointDef();
    jointDef.bodyIdA = first;
    jointDef.bodyIdB = second;
    const b2Vec2 pivot{anchor.x, anchor.y};
    jointDef.localAnchorA = b2Body_GetLocalPoint(first, pivot);
    jointDef.localAnchorB = b2Body_GetLocalPoint(second, pivot);
    joints_.push_back(b2CreateRevoluteJoint(world_, &jointDef));
}

void Body::draw() const {
    static const GLuint program = createBodyShader();
    static GLuint vao{};
    static GLuint vbo{};

    if (program == 0) {
        return;
    }

    std::vector<glm::vec2> vertices;
    vertices.reserve(parts_.size() * 2);
    for (const Part& part : parts_) {
        const b2Vec2 position = b2Body_GetPosition(part.body);
        const b2Rot rotation = b2Body_GetRotation(part.body);
        const float angle = b2Rot_GetAngle(rotation);
        const glm::vec2 axis{glm::cos(angle), glm::sin(angle)};
        const glm::vec2 center = toScreen(position);
        const float length = part.halfSize.x * pixelsPerMeter;
        vertices.push_back(center - axis * length);
        vertices.push_back(center + axis * length);
    }

    if (vao == 0) {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                              sizeof(glm::vec2), nullptr);
        glEnableVertexAttribArray(0);
    }

    glUseProgram(program);
    glUniform4f(glGetUniformLocation(program, "bodyColor"),
                0.2f, 0.75f, 1.0f, 1.0f);
    glUniform1f(glGetUniformLocation(program, "pointSize"), 1.0f);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(vertices.size() * sizeof(glm::vec2)),
                 vertices.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(vertices.size()));
}
