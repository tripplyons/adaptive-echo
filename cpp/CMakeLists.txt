cmake_minimum_required(VERSION 3.14)
project(adaptive_echo VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch pocketfft for FFT-based resampling (scipy.signal.resample parity)
include(FetchContent)
FetchContent_Declare(
    pocketfft
    GIT_REPOSITORY https://github.com/mreineck/pocketfft.git
    GIT_TAG cpp
)
FetchContent_MakeAvailable(pocketfft)

# Find OpenMP
# On macOS with Apple Clang, we need to help CMake find libomp
if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Try to find libomp from Homebrew
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
    elseif(EXISTS "/usr/local/opt/libomp")
        set(OpenMP_ROOT "/usr/local/opt/libomp")
    endif()
    
    if(OpenMP_ROOT)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${OpenMP_ROOT}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${OpenMP_ROOT}/lib/libomp.dylib")
    endif()
endif()

find_package(OpenMP)

# Detect CPU features and set optimization flags
include(CheckCXXCompilerFlag)

# Enable aggressive optimizations
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -funroll-loops -fomit-frame-pointer")

# Try to enable CPU-specific optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i[3-6]86")
    # x86/x86_64 architecture
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
    endif()
    
    # Check for specific SIMD instruction sets
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
    check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE42)
    
    if(COMPILER_SUPPORTS_AVX2)
        message(STATUS "AVX2 support enabled")
    elseif(COMPILER_SUPPORTS_AVX)
        message(STATUS "AVX support enabled")
    elseif(COMPILER_SUPPORTS_SSE42)
        message(STATUS "SSE4.2 support enabled")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64|AARCH64")
    # ARM64 architecture (Apple Silicon, etc.)
    check_cxx_compiler_flag("-mcpu=native" COMPILER_SUPPORTS_MCPU_NATIVE)
    if(COMPILER_SUPPORTS_MCPU_NATIVE)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=native")
    endif()
    message(STATUS "ARM64 architecture detected")
endif()

# Header-only library
add_library(adaptive_echo INTERFACE)
target_include_directories(adaptive_echo INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${pocketfft_SOURCE_DIR}>
)

# Main executable
add_executable(generate_sound src/main.cpp)
target_link_libraries(generate_sound PRIVATE adaptive_echo)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(generate_sound PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found and enabled")
else()
    message(STATUS "OpenMP not found - continuing without parallelization")
endif()

# Platform-specific options
if(MSVC)
    target_compile_options(generate_sound PRIVATE /W4 /O2 /fp:fast)
else()
    target_compile_options(generate_sound PRIVATE -Wall -Wextra -Wpedantic)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(generate_sound PRIVATE -stdlib=libc++)
    endif()
endif()

# Install targets
install(TARGETS adaptive_echo generate_sound
    EXPORT adaptive_echo-targets
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)

install(EXPORT adaptive_echo-targets
    FILE adaptive_echo-targets.cmake
    NAMESPACE adaptive_echo::
    DESTINATION lib/cmake/adaptive_echo
)
