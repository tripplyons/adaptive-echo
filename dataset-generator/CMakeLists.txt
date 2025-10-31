cmake_minimum_required(VERSION 3.22)

project(AdaptiveEchoDatasetGenerator
    VERSION 0.1.0
    DESCRIPTION "Adaptive Echo Dataset Generator"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(FetchContent)

# -------------------
# Eigen
# -------------------
FetchContent_Declare(
    eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.1
)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen3)

if(NOT TARGET Eigen3::Eigen)
  message(FATAL_ERROR "Eigen3::Eigen target was not created")
endif()

# -------------------
# Autodiff
# -------------------
FetchContent_Declare(
    autodiff-source
    GIT_REPOSITORY https://github.com/autodiff/autodiff.git
    GIT_TAG v1.1.2
)
# Disable tests and Python bindings to avoid Catch2 errors
set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(autodiff-source)

# Interface target for autodiff + Eigen
add_library(autodiff-interface INTERFACE)
target_include_directories(autodiff-interface INTERFACE "${autodiff-source_SOURCE_DIR}/include")
target_link_libraries(autodiff-interface INTERFACE Eigen3::Eigen)

# -------------------
# Dataset-generator executable
# -------------------
add_executable(AdaptiveEcho src/Main.cpp)

# Include plugin headers (adjust path if needed)
target_include_directories(AdaptiveEcho PRIVATE
    "${CMAKE_SOURCE_DIR}/../plugin/src"
)

# Link autodiff + Eigen
target_link_libraries(AdaptiveEcho PRIVATE autodiff-interface)

# -------------------
# Warnings
# -------------------
if(MSVC)
    target_compile_options(AdaptiveEcho PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(AdaptiveEcho PRIVATE -Wall -Wextra -Wpedantic)
endif()

# -------------------
# LTO (Release)
# -------------------
if (ADAPTIVEECHO_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_err)
    if (ipo_supported)
        set_property(TARGET AdaptiveEcho PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
endif()

# -------------------
# macOS bundle ID
# -------------------
if (APPLE)
    set(BUNDLE_ID "org.adaptiveecho.AdaptiveEcho")
    set_target_properties(AdaptiveEcho PROPERTIES
        XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "${BUNDLE_ID}"
    )
endif()

# -------------------
# Install targets
# -------------------
install(TARGETS AdaptiveEcho
    RUNTIME DESTINATION bin
    BUNDLE  DESTINATION .
    LIBRARY DESTINATION lib
)
