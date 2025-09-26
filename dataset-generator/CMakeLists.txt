cmake_minimum_required(VERSION 3.22)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(AdaptiveEchoDatasetGenerator
    VERSION 0.1.0
    DESCRIPTION "Adaptive Echo Dataset Generator"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_executable(AdaptiveEcho src/Main.cpp)

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
