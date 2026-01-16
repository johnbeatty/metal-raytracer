#include <metal_stdlib>
#include "SharedTypes.h"

using namespace metal;

// Chapter 1: Basic Tuple Operations

kernel void tuple_add(device const Tuple* a [[ buffer(0) ]],
                      device const Tuple* b [[ buffer(1) ]],
                      device Tuple* result [[ buffer(2) ]],
                      uint index [[ thread_position_in_grid ]])
{
    result[index].components = a[index].components + b[index].components;
}

kernel void tuple_subtract(device const Tuple* a [[ buffer(0) ]],
                           device const Tuple* b [[ buffer(1) ]],
                           device Tuple* result [[ buffer(2) ]],
                           uint index [[ thread_position_in_grid ]])
{
    result[index].components = a[index].components - b[index].components;
}

kernel void tuple_negate(device const Tuple* a [[ buffer(0) ]],
                         device Tuple* result [[ buffer(1) ]],
                         uint index [[ thread_position_in_grid ]])
{
    result[index].components = -a[index].components;
}

kernel void tuple_multiply_scalar(device const Tuple* a [[ buffer(0) ]],
                                  device const float* scalar [[ buffer(1) ]],
                                  device Tuple* result [[ buffer(2) ]],
                                  uint index [[ thread_position_in_grid ]])
{
    result[index].components = a[index].components * scalar[index];
}

kernel void tuple_divide_scalar(device const Tuple* a [[ buffer(0) ]],
                                device const float* scalar [[ buffer(1) ]],
                                device Tuple* result [[ buffer(2) ]],
                                uint index [[ thread_position_in_grid ]])
{
    result[index].components = a[index].components / scalar[index];
}

kernel void vector_magnitude(device const Tuple* a [[ buffer(0) ]],
                             device float* result [[ buffer(1) ]],
                             uint index [[ thread_position_in_grid ]])
{
    result[index] = length(a[index].components);
}

kernel void vector_normalize(device const Tuple* a [[ buffer(0) ]],
                             device Tuple* result [[ buffer(1) ]],
                             uint index [[ thread_position_in_grid ]])
{
    result[index].components = normalize(a[index].components);
}

kernel void vector_dot(device const Tuple* a [[ buffer(0) ]],
                       device const Tuple* b [[ buffer(1) ]],
                       device float* result [[ buffer(2) ]],
                       uint index [[ thread_position_in_grid ]])
{
    result[index] = dot(a[index].components, b[index].components);
}

kernel void vector_cross(device const Tuple* a [[ buffer(0) ]],
                         device const Tuple* b [[ buffer(1) ]],
                         device Tuple* result [[ buffer(2) ]],
                         uint index [[ thread_position_in_grid ]])
{
    // Cross product is only defined for 3-component vectors.
    // We ignore w component or treat it as 0 for direction.
    float3 a3 = a[index].components.xyz;
    float3 b3 = b[index].components.xyz;
    result[index].components = float4(cross(a3, b3), 0.0);
}
