#ifndef SharedTypes_h
#define SharedTypes_h

#include <simd/simd.h>

typedef struct {
    vector_float4 components;
} Tuple;

// Constants for types
#define TUPLE_POINT 1.0
#define TUPLE_VECTOR 0.0

typedef struct {
    vector_float4 position;
    vector_float2 textureCoordinate;
} Vertex;

// Matrix type for 4x4 transformations (Chapter 3)
// Uses column-major order to match Metal/OpenGL conventions
typedef struct {
    vector_float4 columns[4];
} Matrix4x4;

// Identity matrix constant
#define MATRIX4X4_IDENTITY (Matrix4x4){{ \
    {1.0, 0.0, 0.0, 0.0}, \
    {0.0, 1.0, 0.0, 0.0}, \
    {0.0, 0.0, 1.0, 0.0}, \
    {0.0, 0.0, 0.0, 1.0} \
}}

// Chapter 4: Transformation Matrix Functions
// These create 4x4 transformation matrices

// Translation matrix - moves points by (x, y, z)
static inline Matrix4x4 matrix_translation(float x, float y, float z) {
    return (Matrix4x4){{
        {1.0, 0.0, 0.0, 0.0},  // Column 0
        {0.0, 1.0, 0.0, 0.0},  // Column 1
        {0.0, 0.0, 1.0, 0.0},  // Column 2
        {x,   y,   z,   1.0}   // Column 3 (translation)
    }};
}

// Scaling matrix - scales by (x, y, z)
static inline Matrix4x4 matrix_scaling(float x, float y, float z) {
    return (Matrix4x4){{
        {x,   0.0, 0.0, 0.0},  // Column 0
        {0.0, y,   0.0, 0.0},  // Column 1
        {0.0, 0.0, z,   0.0},  // Column 2
        {0.0, 0.0, 0.0, 1.0}   // Column 3
    }};
}

// Rotation around X axis (angle in radians)
static inline Matrix4x4 matrix_rotation_x(float radians) {
    float c = cosf(radians);
    float s = sinf(radians);
    return (Matrix4x4){{
        {1.0, 0.0, 0.0, 0.0},  // Column 0
        {0.0, c,   s,   0.0},  // Column 1
        {0.0, -s,  c,   0.0},  // Column 2
        {0.0, 0.0, 0.0, 1.0}   // Column 3
    }};
}

// Rotation around Y axis (angle in radians)
static inline Matrix4x4 matrix_rotation_y(float radians) {
    float c = cosf(radians);
    float s = sinf(radians);
    return (Matrix4x4){{
        {c,   0.0, -s,  0.0},  // Column 0
        {0.0, 1.0, 0.0, 0.0},  // Column 1
        {s,   0.0, c,   0.0},  // Column 2
        {0.0, 0.0, 0.0, 1.0}   // Column 3
    }};
}

// Rotation around Z axis (angle in radians)
static inline Matrix4x4 matrix_rotation_z(float radians) {
    float c = cosf(radians);
    float s = sinf(radians);
    return (Matrix4x4){{
        {c,   s,   0.0, 0.0},  // Column 0
        {-s,  c,   0.0, 0.0},  // Column 1
        {0.0, 0.0, 1.0, 0.0},  // Column 2
        {0.0, 0.0, 0.0, 1.0}   // Column 3
    }};
}

// Shearing matrix - each coordinate changed in proportion to other two
// xy: x changes in proportion to y
// xz: x changes in proportion to z
// yx: y changes in proportion to x
// yz: y changes in proportion to z
// zx: z changes in proportion to x
// zy: z changes in proportion to y
static inline Matrix4x4 matrix_shearing(float xy, float xz, float yx, float yz, float zx, float zy) {
    return (Matrix4x4){{
        {1.0, yx,  zx,  0.0},  // Column 0
        {xy,  1.0, zy,  0.0},  // Column 1
        {xz,  yz,  1.0, 0.0},  // Column 2
        {0.0, 0.0, 0.0, 1.0}   // Column 3
    }};
}

#endif /* SharedTypes_h */
