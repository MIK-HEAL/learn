cmake_minimum_required(VERSION 3.25)

project(LearnOpenGL
    VERSION 0.1.0
    DESCRIPTION "LearnOpenGL window examples"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(box2d CONFIG REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# ---------------------------------------------------------------- 物理层（无渲染）
# 只包含 Box2D + glm，不链接任何图形库，因此能被 Python 直接以 ctypes 加载。
add_library(rl_env SHARED
    src/ragdoll.cpp
    src/env_api.cpp
)

target_link_libraries(rl_env PRIVATE
    box2d::box2d
    glm::glm
)

target_include_directories(rl_env PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    # 统一使用 Release CRT（/MD），这样即使配置的是 Debug，
    # python.exe 也能安全加载 rl_env.dll。
    set_target_properties(rl_env PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    target_compile_options(rl_env PRIVATE /W4 /utf-8)
else()
    target_compile_options(rl_env PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------- OpenGL 可视化
add_executable(window
    src/main.cpp
    src/body.cpp
    src/console.cpp
    src/ragdoll.cpp
    src/policy.cpp
    src/ground.cpp/ground.cpp
    src/window/window.cpp
)

target_link_libraries(window PRIVATE
    glfw
    GLEW::GLEW
    glm::glm
    box2d::box2d
    opengl32
)

target_include_directories(window PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(window PRIVATE /W4 /utf-8)
else()
    target_compile_options(window PRIVATE -Wall -Wextra -Wpedantic)
endif()
