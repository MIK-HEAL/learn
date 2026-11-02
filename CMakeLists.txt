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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

add_executable(window
    src/main.cpp
    src/ground.cpp/ground.cpp
    src/window/window.cpp
)

target_link_libraries(window PRIVATE
    glfw
    GLEW::GLEW
    glm::glm
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
