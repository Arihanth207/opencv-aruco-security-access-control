cmake_minimum_required(VERSION 3.10)
project(ArUcoAccessControl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(aruco_access src/main.cpp)
target_link_libraries(aruco_access ${OpenCV_LIBS})
