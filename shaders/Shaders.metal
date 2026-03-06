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


// ---------------------------
// Chapter 3: Matrix Operations (Compute)
// ---------------------------

// Helper function to multiply two 4x4 matrices in Metal
float4x4 matrix_multiply(float4x4 a, float4x4 b)
{
    return a * b;
}

kernel void matrix_multiply_kernel(device const Matrix4x4* a [[ buffer(0) ]],
                                   device const Matrix4x4* b [[ buffer(1) ]],
                                   device Matrix4x4* result [[ buffer(2) ]],
                                   uint index [[ thread_position_in_grid ]])
{
    float4x4 ma = float4x4(a[index].columns[0], a[index].columns[1], 
                           a[index].columns[2], a[index].columns[3]);
    float4x4 mb = float4x4(b[index].columns[0], b[index].columns[1], 
                           b[index].columns[2], b[index].columns[3]);
    float4x4 mr = ma * mb;
    result[index].columns[0] = mr[0];
    result[index].columns[1] = mr[1];
    result[index].columns[2] = mr[2];
    result[index].columns[3] = mr[3];
}

kernel void matrix_multiply_tuple(device const Matrix4x4* m [[ buffer(0) ]],
                                  device const Tuple* t [[ buffer(1) ]],
                                  device Tuple* result [[ buffer(2) ]],
                                  uint index [[ thread_position_in_grid ]])
{
    float4x4 mat = float4x4(m[index].columns[0], m[index].columns[1], 
                            m[index].columns[2], m[index].columns[3]);
    result[index].components = mat * t[index].components;
}

kernel void matrix_transpose(device const Matrix4x4* m [[ buffer(0) ]],
                             device Matrix4x4* result [[ buffer(1) ]],
                             uint index [[ thread_position_in_grid ]])
{
    float4x4 mat = float4x4(m[index].columns[0], m[index].columns[1], 
                            m[index].columns[2], m[index].columns[3]);
    float4x4 transposed = transpose(mat);
    result[index].columns[0] = transposed[0];
    result[index].columns[1] = transposed[1];
    result[index].columns[2] = transposed[2];
    result[index].columns[3] = transposed[3];
}

// Helper: Calculate determinant of 4x4 matrix for inverse computation
// Based on Laplace expansion method
float4x4 matrix_inverse_4x4(float4x4 m)
{
    float4x4 inv;
    
    // Calculate cofactors and determinant using expansion by minors
    // For a 4x4 matrix, we expand along first row
    
    float m00 = m[0][0], m01 = m[1][0], m02 = m[2][0], m03 = m[3][0];
    float m10 = m[0][1], m11 = m[1][1], m12 = m[2][1], m13 = m[3][1];
    float m20 = m[0][2], m21 = m[1][2], m22 = m[2][2], m23 = m[3][2];
    float m30 = m[0][3], m31 = m[1][3], m32 = m[2][3], m33 = m[3][3];
    
    // Calculate 3x3 minors for first row cofactors
    float minor00 = m11*(m22*m33 - m23*m32) - m12*(m21*m33 - m23*m31) + m13*(m21*m32 - m22*m31);
    float minor01 = m10*(m22*m33 - m23*m32) - m12*(m20*m33 - m23*m30) + m13*(m20*m32 - m22*m30);
    float minor02 = m10*(m21*m33 - m23*m31) - m11*(m20*m33 - m23*m30) + m13*(m20*m31 - m21*m30);
    float minor03 = m10*(m21*m32 - m22*m31) - m11*(m20*m32 - m22*m30) + m12*(m20*m31 - m21*m30);
    
    float det = m00*minor00 - m01*minor01 + m02*minor02 - m03*minor03;
    
    // If determinant is near zero, return identity as fallback
    if (abs(det) < 1e-6) {
        return float4x4(1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0);
    }
    
    float invDet = 1.0 / det;
    
    // Calculate all cofactors (transposed adjugate)
    inv[0][0] =  (m11*(m22*m33 - m23*m32) - m12*(m21*m33 - m23*m31) + m13*(m21*m32 - m22*m31)) * invDet;
    inv[0][1] = -(m10*(m22*m33 - m23*m32) - m12*(m20*m33 - m23*m30) + m13*(m20*m32 - m22*m30)) * invDet;
    inv[0][2] =  (m10*(m21*m33 - m23*m31) - m11*(m20*m33 - m23*m30) + m13*(m20*m31 - m21*m30)) * invDet;
    inv[0][3] = -(m10*(m21*m32 - m22*m31) - m11*(m20*m32 - m22*m30) + m12*(m20*m31 - m21*m30)) * invDet;
    
    inv[1][0] = -(m01*(m22*m33 - m23*m32) - m02*(m21*m33 - m23*m31) + m03*(m21*m32 - m22*m31)) * invDet;
    inv[1][1] =  (m00*(m22*m33 - m23*m32) - m02*(m20*m33 - m23*m30) + m03*(m20*m32 - m22*m30)) * invDet;
    inv[1][2] = -(m00*(m21*m33 - m23*m31) - m01*(m20*m33 - m23*m30) + m03*(m20*m31 - m21*m30)) * invDet;
    inv[1][3] =  (m00*(m21*m32 - m22*m31) - m01*(m20*m32 - m22*m30) + m02*(m20*m31 - m21*m30)) * invDet;
    
    inv[2][0] =  (m01*(m12*m33 - m13*m32) - m02*(m11*m33 - m13*m31) + m03*(m11*m32 - m12*m31)) * invDet;
    inv[2][1] = -(m00*(m12*m33 - m13*m32) - m02*(m10*m33 - m13*m30) + m03*(m10*m32 - m12*m30)) * invDet;
    inv[2][2] =  (m00*(m11*m33 - m13*m31) - m01*(m10*m33 - m13*m30) + m03*(m10*m31 - m11*m30)) * invDet;
    inv[2][3] = -(m00*(m11*m32 - m12*m31) - m01*(m10*m32 - m12*m30) + m02*(m10*m31 - m11*m30)) * invDet;
    
    inv[3][0] = -(m01*(m12*m23 - m13*m22) - m02*(m11*m23 - m13*m21) + m03*(m11*m22 - m12*m21)) * invDet;
    inv[3][1] =  (m00*(m12*m23 - m13*m22) - m02*(m10*m23 - m13*m20) + m03*(m10*m22 - m12*m20)) * invDet;
    inv[3][2] = -(m00*(m11*m23 - m13*m21) - m01*(m10*m23 - m13*m20) + m03*(m10*m21 - m11*m20)) * invDet;
    inv[3][3] =  (m00*(m11*m22 - m12*m21) - m01*(m10*m22 - m12*m20) + m02*(m10*m21 - m11*m20)) * invDet;
    
    return inv;
}

kernel void matrix_inverse(device const Matrix4x4* m [[ buffer(0) ]],
                           device Matrix4x4* result [[ buffer(1) ]],
                           device bool* success [[ buffer(2) ]],
                           uint index [[ thread_position_in_grid ]])
{
    float4x4 mat = float4x4(m[index].columns[0], m[index].columns[1], 
                            m[index].columns[2], m[index].columns[3]);
    float4x4 inv = matrix_inverse_4x4(mat);
    result[index].columns[0] = inv[0];
    result[index].columns[1] = inv[1];
    result[index].columns[2] = inv[2];
    result[index].columns[3] = inv[3];
    
    // Check if inverse succeeded by verifying A * A^-1 = I (approximately)
    float4x4 product = mat * inv;
    float4 identity_col0 = float4(1.0, 0.0, 0.0, 0.0);
    float4 identity_col1 = float4(0.0, 1.0, 0.0, 0.0);
    float4 identity_col2 = float4(0.0, 0.0, 1.0, 0.0);
    float4 identity_col3 = float4(0.0, 0.0, 0.0, 1.0);
    
    float error = length(product[0] - identity_col0) + length(product[1] - identity_col1) +
                  length(product[2] - identity_col2) + length(product[3] - identity_col3);
    success[index] = (error < 0.01);
}
