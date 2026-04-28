cmake_minimum_required(VERSION 3.16)
project(HFB_Hubbard CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG")

# Find required packages
find_package(Eigen3 REQUIRED CONFIG)
find_package(HDF5 REQUIRED COMPONENTS C)

# Create the zero-temperature executable
add_executable(hfb 
    hfb_numeric_current.cpp 
    data_output.cpp
)

# Create the finite-temperature executable
add_executable(fthfb
    fthfb_numeric.cpp
    data_output.cpp
)

# Link libraries
target_link_libraries(hfb PRIVATE 
    Eigen3::Eigen 
    ${HDF5_LIBRARIES}
)

target_link_libraries(fthfb PRIVATE
    Eigen3::Eigen
    ${HDF5_LIBRARIES}
)

# Include directories
target_include_directories(hfb PRIVATE 
    ${HDF5_INCLUDE_DIRS}
)

target_include_directories(fthfb PRIVATE
    ${HDF5_INCLUDE_DIRS}
)
