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

// Chapter 6: Light and Shading

// Material type: describes how a surface reflects light
typedef struct {
    vector_float4 color;       // Surface color (RGB)
    float ambient;              // Ambient reflection coefficient (0-1)
    float diffuse;              // Diffuse reflection coefficient (0-1)
    float specular;             // Specular reflection coefficient (0-1)
    float shininess;            // Shininess (higher = smaller, tighter specular highlight)
} Material;

// Default material constant
#define DEFAULT_MATERIAL (Material){ \
    (vector_float4){1.0, 1.0, 1.0, 1.0},  /* color: white */ \
    0.1f,                                  /* ambient */ \
    0.9f,                                  /* diffuse */ \
    0.9f,                                  /* specular */ \
    200.0f                                 /* shininess */ \
}

// PointLight type: a light source at a point in space
typedef struct {
    vector_float4 position;    // Light position (point)
    vector_float4 intensity;   // Light color/intensity (RGB)
} PointLight;

#if !METAL_AVAILABLE
// Helper functions for CPU/Objective-C only

// Helper: Create a material with specific properties
static inline Material material_create(vector_float4 color, float ambient, float diffuse, float specular, float shininess) {
    Material m;
    m.color = color;
    m.ambient = ambient;
    m.diffuse = diffuse;
    m.specular = specular;
    m.shininess = shininess;
    return m;
}

// Helper: Create a point light
static inline PointLight point_light_create(vector_float4 position, vector_float4 intensity) {
    PointLight light;
    light.position = position;
    light.intensity = intensity;
    return light;
}

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

// Chapter 6: Surface normals and lighting

// Helper: Compute normal vector at a point on a sphere
// The normal is the vector from the sphere's center to the point
static inline vector_float4 sphere_normal_at(Sphere sphere, vector_float4 point) {
    // Transform point to object space
    matrix_float4x4 inv = matrix_from_columns(sphere.inverseTransform.columns[0],
                                               sphere.inverseTransform.columns[1],
                                               sphere.inverseTransform.columns[2],
                                               sphere.inverseTransform.columns[3]);
    vector_float4 object_point = matrix_multiply(inv, point);
    
    // Compute normal in object space (unit sphere at origin)
    vector_float4 object_normal = object_point;
    object_normal.w = 0.0;  // Make it a vector
    
    // Transform normal back to world space
    // For normals, we use the transpose of the inverse of the upper-left 3x3
    // But for spheres with only translation/rotation/uniform scaling, 
    // we can use the inverse transform and re-normalize
    matrix_float4x4 inv_transpose = matrix_transpose(inv);
    vector_float4 world_normal = matrix_multiply(inv_transpose, object_normal);
    world_normal.w = 0.0;  // Ensure it's a vector
    
    return simd_normalize(world_normal);
}

// Helper: Compute lighting at a point using Phong reflection model
// eye_vector: vector from point to eye
// normal: surface normal at point
// light: point light source
// material: surface material properties
static inline vector_float4 lighting(Material material, PointLight light, 
                                      vector_float4 point, vector_float4 eye_vector, 
                                      vector_float4 normal) {
    // Combine material color with light intensity
    vector_float4 effective_color = material.color * light.intensity;
    effective_color.w = 1.0;  // Ensure color stays as a point/tuple with w=1
    
    // Find direction to light
    vector_float4 light_vector = simd_normalize(light.position - point);
    
    // Compute ambient contribution
    vector_float4 ambient = effective_color * material.ambient;
    ambient.w = 1.0;
    
    // light_dot_normal represents cosine of angle between light vector and normal
    // Negative means light is on opposite side of surface
    float light_dot_normal = simd_dot(light_vector, normal);
    
    vector_float4 diffuse = {0, 0, 0, 0};
    vector_float4 specular = {0, 0, 0, 0};
    
    if (light_dot_normal >= 0) {
        // Compute diffuse contribution
        diffuse = effective_color * material.diffuse * light_dot_normal;
        diffuse.w = 1.0;
        
        // Compute reflection vector
        vector_float4 reflect_vector = -light_vector + normal * 2.0 * light_dot_normal;
        reflect_vector.w = 0.0;
        reflect_vector = simd_normalize(reflect_vector);
        
        // reflect_dot_eye represents cosine of angle between reflection vector and eye
        float reflect_dot_eye = simd_dot(reflect_vector, eye_vector);
        
        if (reflect_dot_eye > 0) {
            // Compute specular contribution
            float factor = powf(reflect_dot_eye, material.shininess);
            specular = light.intensity * material.specular * factor;
            specular.w = 1.0;
        }
    }
    
    // Add all three contributions
    vector_float4 result = ambient + diffuse + specular;
    result.w = 1.0;  // Ensure result is a color point with w=1
    return result;
}

#endif /* !METAL_AVAILABLE */

#endif /* SharedTypes_h */
