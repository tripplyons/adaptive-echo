cmake_minimum_required(VERSION 3.22)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(AdaptiveEcho
    VERSION 0.1.0
    DESCRIPTION "Adaptive Echo - Audio Plugin"
    LANGUAGES C CXX
)

# Set C++ ABI compatibility for libtorch
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

set(BUILD_PYTHON OFF)

if(UNIX)
    # On Unix systems, use 'which' to find clang paths
    execute_process(
        COMMAND which clang
        OUTPUT_VARIABLE CLANG_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE CLANG_RESULT
    )

    execute_process(
        COMMAND which clang++
        OUTPUT_VARIABLE CLANGXX_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE CLANGXX_RESULT
    )

    # Set compiler if command succeeded
    if(CLANG_RESULT EQUAL 0 AND NOT CLANG_OUTPUT STREQUAL "")
        set(CMAKE_C_COMPILER ${CLANG_OUTPUT} CACHE STRING "" FORCE)
        message(STATUS "Found C compiler: ${CMAKE_C_COMPILER}")
    else()
        message(FATAL_ERROR "clang not found via 'which' command - clang is required for this build")
    endif()

    if(CLANGXX_RESULT EQUAL 0 AND NOT CLANGXX_OUTPUT STREQUAL "")
        set(CMAKE_CXX_COMPILER ${CLANGXX_OUTPUT} CACHE STRING "" FORCE)
        message(STATUS "Found CXX compiler: ${CMAKE_CXX_COMPILER}")
    else()
        message(FATAL_ERROR "clang++ not found via 'which' command - clang++ is required for this build")
    endif()
endif()

if(WIN32)
    # Find clang-cl (MSVC-compatible Clang)
    execute_process(
        COMMAND where clang-cl
        OUTPUT_VARIABLE CLANGCL_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE CLANGCL_RESULT
    )
    if(CLANGCL_RESULT EQUAL 0 AND NOT CLANGCL_OUTPUT STREQUAL "")
        string(REGEX MATCH "^[^\r\n]+" CLANGCL_PATH "${CLANGCL_OUTPUT}")
        set(CMAKE_C_COMPILER ${CLANGCL_PATH} CACHE STRING "C compiler" FORCE)
        set(CMAKE_CXX_COMPILER ${CLANGCL_PATH} CACHE STRING "C++ compiler" FORCE)
        message(STATUS "Found MSVC-compatible Clang compiler (clang-cl): ${CMAKE_CXX_COMPILER}")
    else()
        message(FATAL_ERROR "clang-cl not found. Please install LLVM with clang-cl.")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(UNIX AND NOT APPLE)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
    pkg_check_modules(WEBKIT2GTK REQUIRED webkit2gtk-4.1)
endif()

include(FetchContent)
set(CURL_USE_LIBPSL OFF)
set(USE_NGHTTP2 OFF)
set(BROTLI_FOUND OFF)
FetchContent_Declare(curl
    URL https://curl.se/download/curl-8.16.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP true
)
FetchContent_MakeAvailable(curl)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# VST3 SDK location
if(NOT DEFINED VST3_SDK)
    set(VST3_SDK "${CMAKE_SOURCE_DIR}/external/VST3_SDK" CACHE PATH "Path to Steinberg VST3 SDK")
endif()

# Fetch JUCE 8.0.10
# Disable web browser support (not needed for this project)
set(JUCE_WEB_BROWSER 0)
include(FetchContent)
FetchContent_Declare(
    juce
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.10
)
FetchContent_MakeAvailable(juce)

# Fetch Catch2
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_PREFIX_PATH ${Catch2_BINARY_DIR})

# Fetch LibTorch
set(TORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.9.1%2Bcpu.zip")

if(APPLE)
    set(TORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.9.1.zip")
endif()

if(NOT UNIX)
    set(TORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.9.1%2Bcpu.zip")
endif()

include(FetchContent)
FetchContent_Declare(libtorch
    URL ${TORCH_URL}
)
FetchContent_MakeAvailable(libtorch)
find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)
list(APPEND CMAKE_PREFIX_PATH "${libtorch_SOURCE_DIR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(COPY_PLUGIN_AFTER_BUILD TRUE)

if(WIN32)
    set(COPY_PLUGIN_AFTER_BUILD FALSE)
endif()

# Plugin target
juce_add_plugin(AdaptiveEcho
    PRODUCT_NAME "AdaptiveEcho"
    COMPANY_NAME "Adaptive Echo"
    BUNDLE_ID "org.adaptiveecho.AdaptiveEcho"
    PLUGIN_MANUFACTURER_CODE AdEc # 4 chars
    PLUGIN_CODE Ae01 # 4 chars
    FORMATS VST3 AU Standalone
    IS_SYNTH TRUE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD ${COPY_PLUGIN_AFTER_BUILD}
)

# Sources
target_sources(AdaptiveEcho
    PRIVATE
    src/PluginProcessor.cpp
    src/PluginProcessor.hpp
    src/PluginEditor.cpp
    src/PluginEditor.hpp
)

# JUCE modules
target_link_libraries(AdaptiveEcho
    PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_audio_basics
    juce::juce_audio_devices
    juce::juce_dsp
    juce::juce_gui_basics
    juce::juce_graphics
    juce::juce_core
)

# JUCE defines
target_compile_definitions(AdaptiveEcho
    PRIVATE
    JUCE_REPORT_APP_USAGE=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_APPLICATION_VERSION_STRING="${PROJECT_VERSION}"
)

if(MSVC)
    # Get all source files from the target
    get_target_property(SRCS AdaptiveEcho SOURCES)

    foreach(src IN LISTS SRCS)
        # Apply warnings only to your own source files
        if(src MATCHES "${CMAKE_CURRENT_SOURCE_DIR}/src/.*")
            message(STATUS "Applying /W3 to: ${src}")
            set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "/W3")
        else()
            # Suppress warnings for non-project files
            set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "/W0")
        endif()
    endforeach()
else()
    target_compile_options(AdaptiveEcho PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Mark JUCE headers as SYSTEM
target_include_directories(AdaptiveEcho SYSTEM PRIVATE ${JUCE_INCLUDE_DIRS})

# LTO (Release)
if(ADAPTIVEECHO_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_err)

    if(ipo_supported)
        set_property(TARGET AdaptiveEcho PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
endif()

# GTK includes (after target is created)
if(UNIX AND NOT APPLE)
    target_include_directories(AdaptiveEcho SYSTEM PRIVATE ${GTK3_INCLUDE_DIRS} ${WEBKIT2GTK_INCLUDE_DIRS})
endif()

if(APPLE)
    # Set a bundle identifier
    set(BUNDLE_ID "org.adaptiveecho.AdaptiveEcho")
    set_target_properties(AdaptiveEcho PROPERTIES
        XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "${BUNDLE_ID}"
    )
endif()

# Install targets
install(TARGETS AdaptiveEcho
    RUNTIME DESTINATION bin
    BUNDLE DESTINATION .
    LIBRARY DESTINATION lib
)

target_link_options(AdaptiveEcho PRIVATE "LINKER:-as-needed")
target_link_libraries(AdaptiveEcho PRIVATE CURL::libcurl)

juce_generate_juce_header(AdaptiveEcho)

add_executable(AdaptiveEchoDatasetGenerator
    src/DatasetGenerator.cpp
)

target_link_libraries(AdaptiveEchoDatasetGenerator PRIVATE ${TORCH_LIBRARIES})
target_include_directories(AdaptiveEchoDatasetGenerator SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
if(MSVC)
    # Get all source files from the target
    get_target_property(SRCS AdaptiveEchoDatasetGenerator SOURCES)
    foreach(src IN LISTS SRCS)
        # Apply warnings only to your own source files
        if(src MATCHES "${CMAKE_CURRENT_SOURCE_DIR}/src/.*")
            message(STATUS "Applying /W3 to: ${src}")
            set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "/W3")
        else()
            # Suppress warnings for non-project files
            set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "/W0")
        endif()
    endforeach()
else()
    target_compile_options(AdaptiveEchoDatasetGenerator PRIVATE -Wall -Wextra -Wpedantic)
endif()