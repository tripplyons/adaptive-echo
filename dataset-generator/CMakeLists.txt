cmake_minimum_required(VERSION 3.22)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(AdaptiveEchoDatasetGenerator
    VERSION 1.0.0
    DESCRIPTION "Adaptive Echo Dataset Generator"
    LANGUAGES CXX
)

execute_process(
    COMMAND where clang++
    OUTPUT_VARIABLE CLANGXX_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE CLANGXX_RESULT
)

if(CLANGXX_RESULT EQUAL 0 AND NOT CLANGXX_OUTPUT STREQUAL "")
    string(REGEX MATCH "^[^\r\n]+" CLANGXX_PATH "${CLANGXX_OUTPUT}")
    set(CMAKE_CXX_COMPILER ${CLANGXX_PATH} CACHE STRING "" FORCE)
    message(STATUS "Found CXX compiler: ${CMAKE_CXX_COMPILER}")
else()
    message(FATAL_ERROR "clang++ not found via 'where' command - clang++ is required for this build")
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CURL_USE_LIBPSL OFF)
set(USE_NGHTTP2 OFF)

set(CURL_DISABLE_DOCS ON CACHE BOOL "" FORCE)
set(BUILD_CURL_EXE OFF CACHE BOOL "" FORCE) 
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(curl
    URL https://curl.se/download/curl-8.16.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP true
)
FetchContent_MakeAvailable(curl)

# Fetch Catch2
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_PREFIX_PATH ${Catch2_BINARY_DIR})

# Fetch Eigen from official repository
FetchContent_Declare(
    eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.1
)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen3)

if(NOT TARGET Eigen3::Eigen)
  message(FATAL_ERROR "Eigen3::Eigen target was not created")
endif()

set(AUTODIFF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

# Fetch autodiff
FetchContent_Declare(
    autodiff-source
    GIT_REPOSITORY https://github.com/autodiff/autodiff.git
    GIT_TAG v1.1.2
)
FetchContent_MakeAvailable(autodiff-source)
add_library(autodiff-interface INTERFACE)
target_include_directories(autodiff-interface INTERFACE "${autodiff-source_SOURCE_DIR}")
target_link_libraries(autodiff-interface INTERFACE Eigen3::Eigen)

# Add the executable
add_executable(AdaptiveEchoDatasetGenerator src/Main.cpp)

# Warnings
target_compile_options(AdaptiveEchoDatasetGenerator PRIVATE -Wall -Wextra -Wpedantic)

target_link_libraries(AdaptiveEchoDatasetGenerator PRIVATE 
    CURL::libcurl
    autodiff-interface
)