cmake_minimum_required(VERSION 3.15)
project(RayTracerChallenge OBJC CXX)

set(CMAKE_OBJC_STANDARD 11)

find_library(METAL_LIB Metal)
find_library(FOUNDATION_LIB Foundation)

# Metal Shader Compilation
set(SHADER_SRC shaders/Shaders.metal)
include_directories(src)

set(METALLIB ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)

# Simple build rule for a single metal file. 
# For a real project we might want a macro to handle multiple files.
add_custom_command(
    OUTPUT ${METALLIB}
    COMMAND xcrun -sdk macosx metal -I ${CMAKE_SOURCE_DIR}/src -c ${CMAKE_SOURCE_DIR}/${SHADER_SRC} -o ${CMAKE_CURRENT_BINARY_DIR}/Shaders.air
    COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/Shaders.air -o ${METALLIB}
    DEPENDS ${SHADER_SRC}
    COMMENT "Building Metal Library"
)

add_custom_target(MetalShaders DEPENDS ${METALLIB})

add_executable(RayTracer src/main.m)
add_dependencies(RayTracer MetalShaders)

target_link_libraries(RayTracer ${METAL_LIB} ${FOUNDATION_LIB})

# Ensure the metallib is next to the executable (optional, but helpful for finding it)
add_custom_command(TARGET RayTracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${METALLIB}
    $<TARGET_FILE_DIR:RayTracer>/default.metallib)
