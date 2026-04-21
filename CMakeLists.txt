cmake_minimum_required(VERSION 3.24)

project(adaptive_echo VERSION 1.0.0 LANGUAGES C CXX)

include(FetchContent)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS deployment target")

FetchContent_Declare(
    juce
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 29396c22c93392d6738e021b83196283d6e4d850 # 8.0.12
)

set(JUCE_ENABLE_MODULE_SOURCE_GROUPS ON CACHE BOOL "" FORCE)
set(JUCE_COPY_PLUGIN_AFTER_BUILD TRUE CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(juce)

set(POCKETFFT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/cpp/third_party/pocketfft")

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
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

set(ADAPTIVE_ECHO_ENGINE_RELEASE_OPTIONS "")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i[3-6]86")
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        list(APPEND ADAPTIVE_ECHO_ENGINE_RELEASE_OPTIONS -march=native)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64|AARCH64")
    check_cxx_compiler_flag("-mcpu=native" COMPILER_SUPPORTS_MCPU_NATIVE)
    if(COMPILER_SUPPORTS_MCPU_NATIVE)
        list(APPEND ADAPTIVE_ECHO_ENGINE_RELEASE_OPTIONS -mcpu=native)
    endif()
endif()

add_library(adaptive_echo_engine STATIC
    cpp/src/engine.cpp
)

target_include_directories(adaptive_echo_engine
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
        ${POCKETFFT_ROOT}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(adaptive_echo_engine PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(adaptive_echo_engine PRIVATE /W4 /O2 /fp:fast)
else()
    target_compile_options(adaptive_echo_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_options(adaptive_echo_engine
    PRIVATE
        $<$<CONFIG:Release>:-ffast-math;-funroll-loops;-fomit-frame-pointer>
        $<$<CONFIG:Release>:${ADAPTIVE_ECHO_ENGINE_RELEASE_OPTIONS}>
)

juce_add_plugin(AdaptiveEcho
    COMPANY_NAME "Adaptive Echo"
    BUNDLE_ID "com.adaptiveecho.plugin"
    IS_SYNTH TRUE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE
    COPY_PLUGIN_AFTER_BUILD TRUE
    PLUGIN_MANUFACTURER_CODE AdEc
    PLUGIN_CODE AdEo
    FORMATS AU VST3 Standalone
    PRODUCT_NAME "Adaptive Echo"
)

target_sources(AdaptiveEcho
    PRIVATE
        plugin/source/PluginEditor.cpp
        plugin/source/PluginProcessor.cpp
)

target_compile_definitions(AdaptiveEcho
    PRIVATE
        JUCE_IGNORE_VST3_MISMATCHED_PARAMETER_ID_WARNING=1
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
)

target_link_libraries(AdaptiveEcho
    PRIVATE
        adaptive_echo_engine
        juce::juce_audio_formats
        juce::juce_audio_utils
        juce::juce_gui_extra
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

add_executable(compare_optimizers
    cpp/compare_optimizers.cpp
)

target_link_libraries(compare_optimizers
    PRIVATE
        adaptive_echo_engine
        juce::juce_audio_formats
        juce::juce_core
        juce::juce_data_structures
        juce::juce_events
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

add_executable(loss_smoke_test
    cpp/tests/loss_smoke_test.cpp
)

target_link_libraries(loss_smoke_test
    PRIVATE
        adaptive_echo_engine
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)
