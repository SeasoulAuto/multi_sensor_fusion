cmake_minimum_required(VERSION 3.10)
project(KalmanFilter)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Add include directories
include_directories(${EIGEN3_INCLUDE_DIR})

# Create a shared library
add_library(KalmanFilter SHARED kalman_filter.cc)

# Link the library with Eigen3
target_link_libraries(KalmanFilter ${EIGEN3_LIBRARIES})