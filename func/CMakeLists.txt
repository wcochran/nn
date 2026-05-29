cmake_minimum_required(VERSION 3.16)

project(function_approximator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(function_approximator
    main.cpp
    neural_network.cpp
)

target_link_libraries(function_approximator PRIVATE Eigen3::Eigen)

if (MSVC)
    target_compile_options(function_approximator PRIVATE /W4)
else()
    target_compile_options(function_approximator PRIVATE -Wall -Wextra -pedantic)
endif()
