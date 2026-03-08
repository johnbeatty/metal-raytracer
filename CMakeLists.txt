cmake_minimum_required(VERSION 3.15)
project(RayTracerChallenge OBJC CXX)

set(CMAKE_OBJC_STANDARD 11)

find_library(METAL_LIB Metal)
find_library(METALKIT_LIB MetalKit)
find_library(COCOA_LIB Cocoa)
find_library(QUARTZ_LIB QuartzCore)

# Metal Shader Compilation
set(SHADER_SRC shaders/Shaders.metal)
include_directories(src)

set(METALLIB ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)

add_custom_command(
    OUTPUT ${METALLIB}
    COMMAND xcrun -sdk macosx metal -I ${CMAKE_SOURCE_DIR}/src -c ${CMAKE_SOURCE_DIR}/${SHADER_SRC} -o ${CMAKE_CURRENT_BINARY_DIR}/Shaders.air
    COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/Shaders.air -o ${METALLIB}
    DEPENDS ${SHADER_SRC}
    COMMENT "Building Metal Library"
)

add_custom_target(MetalShaders DEPENDS ${METALLIB})

set(SOURCES 
    src/main.m
    src/AppDelegate.m
    src/Renderer.m
)

add_executable(RayTracer MACOSX_BUNDLE ${SOURCES})
add_dependencies(RayTracer MetalShaders)

target_link_libraries(RayTracer 
    ${METAL_LIB} 
    ${METALKIT_LIB}
    ${COCOA_LIB}
    ${QUARTZ_LIB}
)

# Ensure the metallib is copied into the bundle resources
add_custom_command(TARGET RayTracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:RayTracer>/../Resources
    COMMAND ${CMAKE_COMMAND} -E copy
    ${METALLIB}
    $<TARGET_FILE_DIR:RayTracer>/../Resources/default.metallib)

# Enable testing
enable_testing()

# XCTest Framework
find_library(XCTEST_LIB XCTest)

# Test Bundle
set(TEST_SOURCES
    tests/TupleTests.m
    tests/VectorTests.m
    tests/MatrixTests.m
    tests/TransformationTests.m
    tests/IntersectionTests.m
    tests/LightingTests.m
    tests/WorldTests.m
    tests/ShadowTests.m
    tests/PlaneTests.m
    tests/PatternTests.m
    tests/ReflectionRefractionTests.m
    tests/CubeTests.m
    tests/CylinderTests.m
    tests/main.m
)

add_executable(RayTracerTests MACOSX_BUNDLE ${TEST_SOURCES})
target_link_libraries(RayTracerTests
    ${XCTEST_LIB}
    ${METAL_LIB}
    ${METALKIT_LIB}
    ${COCOA_LIB}
    ${QUARTZ_LIB}
)

# Copy metal library to test bundle
add_custom_command(TARGET RayTracerTests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:RayTracerTests>/../Resources
    COMMAND ${CMAKE_COMMAND} -E copy
    ${METALLIB}
    $<TARGET_FILE_DIR:RayTracerTests>/../Resources/default.metallib)

# Add to CTest
add_test(NAME XCTest COMMAND RayTracerTests)
