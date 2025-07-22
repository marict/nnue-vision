cmake_minimum_required(VERSION 3.14)
project(NNUEEngine VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Platform-specific flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # x86_64 specific optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx2 -mfma")
    add_definitions(-D__AVX2__)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    # ARM specific optimizations
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=native")
    endif()
    add_definitions(-D__ARM_NEON__)
endif()

# Warning flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")

# Include directories
include_directories(include)

# Source files
set(NNUE_SOURCES
    src/nnue_engine.cpp
    src/simd_scalar.cpp
    src/simd_avx2.cpp
    src/simd_neon.cpp
)

# Create the main library
add_library(nnue_engine STATIC ${NNUE_SOURCES})

# Set library properties
set_target_properties(nnue_engine PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER include/nnue_engine.h
)

# Link libraries
if(WIN32)
    # Windows specific libraries
    target_link_libraries(nnue_engine)
else()
    # Unix-like systems
    target_link_libraries(nnue_engine pthread)
endif()

# Create test executable
add_executable(test_nnue_engine tests/test_nnue_engine.cpp)
target_link_libraries(test_nnue_engine nnue_engine)

# Create example executable
add_executable(nnue_example examples/example_usage.cpp)
target_link_libraries(nnue_example nnue_engine)

# Installation
install(TARGETS nnue_engine
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

install(TARGETS test_nnue_engine nnue_example
    RUNTIME DESTINATION bin
)

# Testing
enable_testing()
add_test(NAME NNUEEngineTests COMMAND test_nnue_engine)

# CPack configuration for packaging
set(CPACK_PACKAGE_NAME "NNUEEngine")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-performance NNUE neural network engine for visual wake words")
set(CPACK_PACKAGE_VENDOR "NNUE Vision")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/../README.md")

include(CPack)

# Documentation with Doxygen (optional)
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    if(EXISTS ${DOXYGEN_IN})
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        
        add_custom_target(docs ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
    endif()
endif()

# Print configuration summary
message(STATUS "=== NNUE Engine Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS}")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug flags: ${CMAKE_CXX_FLAGS_DEBUG}")
endif()

# Performance hints
message(STATUS "")
message(STATUS "=== Performance Hints ===")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    message(STATUS "AVX2 support enabled for x86_64")
    message(STATUS "Use -DCMAKE_BUILD_TYPE=Release for best performance")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    message(STATUS "NEON support enabled for ARM")
    message(STATUS "Use -DCMAKE_BUILD_TYPE=Release for best performance")
else()
    message(STATUS "Using scalar implementation - consider upgrading to a modern CPU")
endif()
message(STATUS "================================")

# Custom targets for convenience
add_custom_target(run_tests
    COMMAND test_nnue_engine
    DEPENDS test_nnue_engine
    COMMENT "Running NNUE engine tests"
)

add_custom_target(run_example
    COMMAND nnue_example
    DEPENDS nnue_example
    COMMENT "Running NNUE engine example"
)

# Benchmark target (if model file exists)
add_custom_target(benchmark
    COMMAND nnue_example model.nnue
    DEPENDS nnue_example
    COMMENT "Running NNUE engine benchmark (requires model.nnue)"
) 