cmake_minimum_required(VERSION 3.16)

project(mnist_mlp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(mnist_mlp
    main.cpp
    mnist_loader.cpp
    neural_network.cpp
)

target_link_libraries(mnist_mlp PRIVATE Eigen3::Eigen)

target_compile_options(mnist_mlp PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)
