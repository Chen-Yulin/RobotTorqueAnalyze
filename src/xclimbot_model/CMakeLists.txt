cmake_minimum_required(VERSION 3.10.2)
project(xclimbot_model)
find_package(ament_cmake REQUIRED)

ament_package()

install(DIRECTORY meshes urdf DESTINATION share/${PROJECT_NAME})
