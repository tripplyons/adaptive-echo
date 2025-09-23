cmake_minimum_required(VERSION 3.22)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(AdaptiveEcho
    VERSION 0.1.0
    DESCRIPTION "Adaptive Echo - Audio Plugin"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
if(UNIX AND NOT APPLE)
    pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
    pkg_check_modules(WEBKIT2GTK REQUIRED webkit2gtk-4.1)

    include_directories(${GTK3_INCLUDE_DIRS} ${WEBKIT2GTK_INCLUDE_DIRS})
endif()

find_package(CURL REQUIRED)
#target_link_libraries(${PROJECT_NAME} PRIVATE CURL::libcurl)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# VST3 SDK location
if (NOT DEFINED VST3_SDK)
    set(VST3_SDK "${CMAKE_SOURCE_DIR}/external/VST3_SDK" CACHE PATH "Path to Steinberg VST3 SDK")
endif()

# Fetch JUCE 8.0.10
include(FetchContent)
FetchContent_Declare(
    juce
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        8.0.10
)
FetchContent_MakeAvailable(juce)

# Plugin target
juce_add_plugin(AdaptiveEcho
    PRODUCT_NAME               "Adaptive Echo"
    COMPANY_NAME               "Adaptive Echo"                # <- change me
    PLUGIN_MANUFACTURER_CODE   AdEc                         # 4 chars
    PLUGIN_CODE                Ae01                         # 4 chars
    FORMATS                    VST3 AU Standalone
    IS_SYNTH                   TRUE
    NEEDS_MIDI_INPUT           TRUE
    NEEDS_MIDI_OUTPUT          FALSE
    IS_MIDI_EFFECT             FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD    TRUE
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
        JUCE_DISPLAY_SPLASH_SCREEN=0
        JUCE_REPORT_APP_USAGE=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_APPLICATION_VERSION_STRING="${PROJECT_VERSION}"
)

# Warnings
if (MSVC)
    target_compile_options(AdaptiveEcho PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(AdaptiveEcho PRIVATE -Wall -Wextra -Wpedantic)
endif()

# LTO (Release)
if (ADAPTIVEECHO_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_err)
    if (ipo_supported)
        set_property(TARGET AdaptiveEcho PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
endif()

if (APPLE)
    # Set a bundle identifier
    set(BUNDLE_ID "org.adaptiveecho.AdaptiveEcho")
    set_target_properties(AdaptiveEcho PROPERTIES
        XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "${BUNDLE_ID}"
    )
endif()

# Install targets
install(TARGETS AdaptiveEcho
    RUNTIME DESTINATION bin
    BUNDLE  DESTINATION .
    LIBRARY DESTINATION lib
)

target_link_libraries(${PROJECT_NAME} PRIVATE CURL::libcurl)
juce_generate_juce_header(${PROJECT_NAME})