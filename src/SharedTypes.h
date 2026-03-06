#ifndef SharedTypes_h
#define SharedTypes_h

#include <simd/simd.h>

// Metal vs C/Objective-C detection
#ifdef __METAL_VERSION__
    // Metal shading language
    #define METAL_AVAILABLE 1
#else
    // C/Objective-C
    #define METAL_AVAILABLE 0
    #include <math.h>
#endif

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

#if !METAL_AVAILABLE
// Chapter 4: Transformation Matrix Functions
// These create 4x4 transformation matrices (CPU/Objective-C only)

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

#endif /* !METAL_AVAILABLE */

// Chapter 5: Ray-Sphere Intersections
// These types are shared between Metal and CPU

// Ray type: origin (point) + direction (vector)
typedef struct {
    vector_float4 origin;      // w = 1 (point)
    vector_float4 direction;   // w = 0 (vector)
} Ray;

// Intersection type: t value and object reference
// t is the distance along the ray where intersection occurs
typedef struct {
    float t;                   // Distance along ray
    int objectId;              // ID of intersected object (-1 if none)
} Intersection;

// Maximum intersections per ray (for array sizing)
#define MAX_INTERSECTIONS 16

// Sphere type: unit sphere at origin with transform matrix
typedef struct {
    int id;                      // Unique identifier
    Matrix4x4 transform;         // Transform from object space to world space
    Matrix4x4 inverseTransform;  // Cached inverse for ray transformation
} Sphere;

#if !METAL_AVAILABLE
// Helper functions for CPU/Objective-C only

// Helper: Create a ray from origin and direction
static inline Ray ray_create(vector_float4 origin, vector_float4 direction) {
    return (Ray){origin, direction};
}

// Helper: Compute position at distance t along ray
static inline vector_float4 ray_position(Ray ray, float t) {
    return ray.origin + ray.direction * t;
}

// Helper: Transform a ray by a matrix
// Used to transform rays into object space for intersection tests
static inline Ray ray_transform(Ray ray, Matrix4x4 matrix) {
    matrix_float4x4 m = matrix_from_columns(matrix.columns[0], matrix.columns[1], 
                                            matrix.columns[2], matrix.columns[3]);
    Ray result;
    result.origin = matrix_multiply(m, ray.origin);
    result.direction = matrix_multiply(m, ray.direction);
    return result;
}

// Helper: Create a sphere with identity transform
static inline Sphere sphere_create(int id) {
    Sphere s;
    s.id = id;
    s.transform = MATRIX4X4_IDENTITY;
    s.inverseTransform = MATRIX4X4_IDENTITY;
    return s;
}

// Helper: Set sphere transform and compute inverse
static inline void sphere_set_transform(Sphere* sphere, Matrix4x4 transform) {
    sphere->transform = transform;
    matrix_float4x4 mat = matrix_from_columns(transform.columns[0], transform.columns[1], 
                                              transform.columns[2], transform.columns[3]);
    matrix_float4x4 inv = matrix_invert(mat);
    sphere->inverseTransform.columns[0] = inv.columns[0];
    sphere->inverseTransform.columns[1] = inv.columns[1];
    sphere->inverseTransform.columns[2] = inv.columns[2];
    sphere->inverseTransform.columns[3] = inv.columns[3];
}

#endif /* !METAL_AVAILABLE */

#endif /* SharedTypes_h */
