cmake_minimum_required(VERSION 3.16)
project(HFB_Hubbard CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG")

# Find required packages
find_package(Eigen3 REQUIRED CONFIG)
find_package(HDF5 REQUIRED COMPONENTS C)

set(COMMON_LIBS
    Eigen3::Eigen
    ${HDF5_LIBRARIES}
)

set(COMMON_INCLUDES
    ${HDF5_INCLUDE_DIRS}
)

function(add_hfb_executable target_name)
    add_executable(${target_name} ${ARGN})
    target_link_libraries(${target_name} PRIVATE ${COMMON_LIBS})
    target_include_directories(${target_name} PRIVATE ${COMMON_INCLUDES})
endfunction()

# Root zero-temperature executable.
add_hfb_executable(hfb
    hfb_numeric_current.cpp
    data_output.cpp
)

# Finite-temperature executable in ./fthfb.
add_hfb_executable(fthfb
    fthfb/fthfb_numeric.cpp
    fthfb/data_output.cpp
)

# Time-dependent HFB executable in ./tdhfb.
add_hfb_executable(tdhfb
    tdhfb/tdhfb_main.cpp
    tdhfb/tdhfb.cpp
    tdhfb/hfb_numeric_current.cpp
    tdhfb/data_output.cpp
)

# Imaginary-time TDHFB quench executable.
add_hfb_executable(tdhfb_quench
    tdhfb/tdhfb_quench_main.cpp
    tdhfb/tdhfb_quench.cpp
    tdhfb/hfb_numeric_current.cpp
    tdhfb/data_output.cpp
)
