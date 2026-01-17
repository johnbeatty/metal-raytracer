#include <metal_stdlib>
#include "SharedTypes.h"

using namespace metal;

// ---------------------------
// Structs for Vertex Shader
// ---------------------------
struct RasterizerData
{
    // The [[position]] attribute of this member indicates that this value
    // is the clip space position of the vertex when this structure is
    // returned from the vertex function.
    float4 position [[position]];

    // Since this member does not have a special attribute, the rasterizer
    // will interpolate its value with the values of the other triangle vertices
    // and then pass the interpolated value to the fragment shader for each
    // fragment in the triangle.
    float2 textureCoordinate;
};

// ---------------------------
// Vertex Shader
// ---------------------------
vertex RasterizerData
vertexShader(uint vertexID [[vertex_id]],
             constant Vertex *vertices [[buffer(0)]])
{
    RasterizerData out;
    
    out.position = vertices[vertexID].position;
    out.textureCoordinate = vertices[vertexID].textureCoordinate;
    
    return out;
}

// ---------------------------
// Fragment Shader
// ---------------------------
fragment float4 fragmentShader(RasterizerData in [[stage_in]])
{
    // Just return a static color for now to verify the pipeline works
    // We'll hook up a texture later
    return float4(in.textureCoordinate.x, in.textureCoordinate.y, 0.0, 1.0);
}


// ---------------------------
// Existing Tuple Operations (Compute)
// ---------------------------

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
    float3 a3 = a[index].components.xyz;
    float3 b3 = b[index].components.xyz;
    result[index].components = float4(cross(a3, b3), 0.0);
}
