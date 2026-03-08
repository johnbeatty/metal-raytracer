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

// Chapter 6: Light and Shading

// Material type: describes how a surface reflects light
typedef struct {
    vector_float4 color;       // Surface color (RGB)
    float ambient;              // Ambient reflection coefficient (0-1)
    float diffuse;              // Diffuse reflection coefficient (0-1)
    float specular;             // Specular reflection coefficient (0-1)
    float shininess;            // Shininess (higher = smaller, tighter specular highlight)
    // Chapter 11: Reflection and Refraction
    float reflective;           // Reflectivity (0 = no reflection, 1 = perfect mirror)
    float transparency;         // Transparency (0 = opaque, 1 = fully transparent)
    float refractive_index;     // Refractive index (1.0 = vacuum/air, 1.5 = glass, 2.417 = diamond)
} Material;

// Default material constant
#define DEFAULT_MATERIAL (Material){ \
    (vector_float4){1.0, 1.0, 1.0, 1.0},  /* color: white */ \
    0.1f,                                  /* ambient */ \
    0.9f,                                  /* diffuse */ \
    0.9f,                                  /* specular */ \
    200.0f,                                /* shininess */ \
    0.0f,                                  /* reflective */ \
    0.0f,                                  /* transparency */ \
    1.0f                                   /* refractive_index */ \
}

// Sphere type: unit sphere at origin with transform matrix
typedef struct {
    int id;                      // Unique identifier
    Matrix4x4 transform;         // Transform from object space to world space
    Matrix4x4 inverseTransform;  // Cached inverse for ray transformation
    Material material;           // Surface material properties
} Sphere;

// Chapter 9: Planes

// Plane type: infinite plane at y=0 (can be transformed)
typedef struct {
    int id;                      // Unique identifier
    Matrix4x4 transform;         // Transform from object space to world space
    Matrix4x4 inverseTransform;  // Cached inverse for ray transformation
} Plane;

// Chapter 12: Cubes

// Cube type: unit cube centered at origin with faces at x=±1, y=±1, z=±1
typedef struct {
    int id;                      // Unique identifier
    Matrix4x4 transform;         // Transform from object space to world space
    Matrix4x4 inverseTransform;  // Cached inverse for ray transformation
    Material material;           // Surface material properties
} Cube;

// Chapter 13: Cylinders

// Cylinder type: infinite cylinder along y-axis (can be truncated with min/max)
// In object space: x² + z² = 1 (unit radius)
typedef struct {
    int id;                      // Unique identifier
    Matrix4x4 transform;         // Transform from object space to world space
    Matrix4x4 inverseTransform;  // Cached inverse for ray transformation
    Material material;           // Surface material properties
    float minimum;               // Minimum y value (for truncation, -INFINITY for infinite)
    float maximum;               // Maximum y value (for truncation, INFINITY for infinite)
    bool closed;                 // If true, caps are added at minimum and maximum
} Cylinder;

// Chapter 14: Groups

// Shape type enumeration for group children
typedef enum {
    SHAPE_SPHERE = 0,
    SHAPE_PLANE = 1,
    SHAPE_CUBE = 2,
    SHAPE_CYLINDER = 3,
    SHAPE_GROUP = 4  // Groups can contain other groups
} ShapeType;

// Maximum children per group
#define MAX_GROUP_CHILDREN 20

// Group type: a container for multiple shapes with a shared transform
typedef struct {
    int id;                      // Unique identifier
    Matrix4x4 transform;         // Transform applied to all children
    Matrix4x4 inverseTransform;  // Cached inverse for ray transformation
    int child_ids[MAX_GROUP_CHILDREN];      // IDs of child shapes
    ShapeType child_types[MAX_GROUP_CHILDREN]; // Types of child shapes
    int child_count;             // Number of children
    Material material;           // Default material (children can override)
} Group;

// Pattern type enumeration
typedef enum {
    PATTERN_STRIPE = 0,
    PATTERN_GRADIENT = 1,
    PATTERN_RING = 2,
    PATTERN_CHECKER = 3,
    PATTERN_SOLID = 4  // No pattern, just use color a
} PatternType;

// SurfacePattern struct: defines a pattern for use in materials
// Note: Named SurfacePattern to avoid conflict with macOS QuickDraw Pattern type
typedef struct {
    PatternType type;              // Type of pattern
    vector_float4 a;               // First color
    vector_float4 b;               // Second color
    Matrix4x4 transform;           // Pattern transformation matrix
    Matrix4x4 inverseTransform;    // Cached inverse of transform
} SurfacePattern;

#if !METAL_AVAILABLE
// Helper: Create a stripe pattern (alternates in X)
static inline SurfacePattern pattern_stripe(vector_float4 a, vector_float4 b) {
    SurfacePattern p;
    p.type = PATTERN_STRIPE;
    p.a = a;
    p.b = b;
    p.transform = MATRIX4X4_IDENTITY;
    p.inverseTransform = MATRIX4X4_IDENTITY;
    return p;
}

// Helper: Create a gradient pattern (linear blend in X)
static inline SurfacePattern pattern_gradient(vector_float4 a, vector_float4 b) {
    SurfacePattern p;
    p.type = PATTERN_GRADIENT;
    p.a = a;
    p.b = b;
    p.transform = MATRIX4X4_IDENTITY;
    p.inverseTransform = MATRIX4X4_IDENTITY;
    return p;
}

// Helper: Create a ring pattern (concentric circles in XZ plane)
static inline SurfacePattern pattern_ring(vector_float4 a, vector_float4 b) {
    SurfacePattern p;
    p.type = PATTERN_RING;
    p.a = a;
    p.b = b;
    p.transform = MATRIX4X4_IDENTITY;
    p.inverseTransform = MATRIX4X4_IDENTITY;
    return p;
}

// Helper: Create a checker pattern (3D checkers)
static inline SurfacePattern pattern_checker(vector_float4 a, vector_float4 b) {
    SurfacePattern p;
    p.type = PATTERN_CHECKER;
    p.a = a;
    p.b = b;
    p.transform = MATRIX4X4_IDENTITY;
    p.inverseTransform = MATRIX4X4_IDENTITY;
    return p;
}

// Helper: Set pattern transform and compute inverse
static inline void pattern_set_transform(SurfacePattern* pattern, Matrix4x4 transform) {
    pattern->transform = transform;
    matrix_float4x4 mat = matrix_from_columns(transform.columns[0], transform.columns[1],
                                               transform.columns[2], transform.columns[3]);
    matrix_float4x4 inv = matrix_invert(mat);
    pattern->inverseTransform.columns[0] = inv.columns[0];
    pattern->inverseTransform.columns[1] = inv.columns[1];
    pattern->inverseTransform.columns[2] = inv.columns[2];
    pattern->inverseTransform.columns[3] = inv.columns[3];
}

// Helper: Evaluate stripe pattern at a point
static inline vector_float4 pattern_stripe_at(SurfacePattern pattern, vector_float4 point) {
    // Alternates between a and b based on X coordinate
    // Uses symmetric alternating pattern: [-1,0) mirrors [0,1), [-2,-1) mirrors [1,2), etc.
    int stripe_index = (int)floorf(fabsf(point.x));
    if (stripe_index % 2 == 0) {
        return pattern.a;
    } else {
        return pattern.b;
    }
}

// Helper: Evaluate gradient pattern at a point
static inline vector_float4 pattern_gradient_at(SurfacePattern pattern, vector_float4 point) {
    // Linear interpolation between a and b based on X coordinate
    float distance = point.x - floorf(point.x);
    vector_float4 result = pattern.a + (pattern.b - pattern.a) * distance;
    result.w = 1.0;
    return result;
}

// Helper: Evaluate ring pattern at a point
static inline vector_float4 pattern_ring_at(SurfacePattern pattern, vector_float4 point) {
    // Alternates based on distance from Y axis
    float distance = sqrtf(point.x * point.x + point.z * point.z);
    int ring_index = (int)floorf(distance);
    if (ring_index % 2 == 0) {
        return pattern.a;
    } else {
        return pattern.b;
    }
}

// Helper: Evaluate checker pattern at a point
static inline vector_float4 pattern_checker_at(SurfacePattern pattern, vector_float4 point) {
    // 3D checkerboard pattern
    int x_index = (int)floorf(point.x);
    int y_index = (int)floorf(point.y);
    int z_index = (int)floorf(point.z);
    
    if ((x_index + y_index + z_index) % 2 == 0) {
        return pattern.a;
    } else {
        return pattern.b;
    }
}

// Helper: Evaluate a pattern at a point (in pattern space)
static inline vector_float4 pattern_at(SurfacePattern pattern, vector_float4 point) {
    // Transform point to pattern space
    matrix_float4x4 inv = matrix_from_columns(pattern.inverseTransform.columns[0],
                                               pattern.inverseTransform.columns[1],
                                               pattern.inverseTransform.columns[2],
                                               pattern.inverseTransform.columns[3]);
    vector_float4 pattern_point = matrix_multiply(inv, point);
    
    switch (pattern.type) {
        case PATTERN_STRIPE:
            return pattern_stripe_at(pattern, pattern_point);
        case PATTERN_GRADIENT:
            return pattern_gradient_at(pattern, pattern_point);
        case PATTERN_RING:
            return pattern_ring_at(pattern, pattern_point);
        case PATTERN_CHECKER:
            return pattern_checker_at(pattern, pattern_point);
        case PATTERN_SOLID:
        default:
            return pattern.a;
    }
}

// Helper: Get pattern color at a point on an object (includes object transform)
// This transforms the point from world space to object space, then evaluates pattern
static inline vector_float4 pattern_at_object(SurfacePattern pattern, Matrix4x4 object_inverse_transform, vector_float4 world_point) {
    // First transform from world to object space
    matrix_float4x4 obj_inv = matrix_from_columns(object_inverse_transform.columns[0],
                                                   object_inverse_transform.columns[1],
                                                   object_inverse_transform.columns[2],
                                                   object_inverse_transform.columns[3]);
    vector_float4 object_point = matrix_multiply(obj_inv, world_point);
    
    // Then evaluate pattern at that point
    return pattern_at(pattern, object_point);
}
#endif /* !METAL_AVAILABLE */

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
    s.material = DEFAULT_MATERIAL;
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

// Chapter 12: Cube helper functions

// Helper: Create a cube with identity transform
static inline Cube cube_create(int id) {
    Cube c;
    c.id = id;
    c.transform = MATRIX4X4_IDENTITY;
    c.inverseTransform = MATRIX4X4_IDENTITY;
    c.material = DEFAULT_MATERIAL;
    return c;
}

// Helper: Set cube transform and compute inverse
static inline void cube_set_transform(Cube* cube, Matrix4x4 transform) {
    cube->transform = transform;
    matrix_float4x4 mat = matrix_from_columns(transform.columns[0], transform.columns[1], 
                                              transform.columns[2], transform.columns[3]);
    matrix_float4x4 inv = matrix_invert(mat);
    cube->inverseTransform.columns[0] = inv.columns[0];
    cube->inverseTransform.columns[1] = inv.columns[1];
    cube->inverseTransform.columns[2] = inv.columns[2];
    cube->inverseTransform.columns[3] = inv.columns[3];
}

// Chapter 13: Cylinder helper functions

// Helper: Create a cylinder with identity transform
// Default is an infinite cylinder along y-axis with radius 1
static inline Cylinder cylinder_create(int id) {
    Cylinder c;
    c.id = id;
    c.transform = MATRIX4X4_IDENTITY;
    c.inverseTransform = MATRIX4X4_IDENTITY;
    c.material = DEFAULT_MATERIAL;
    c.minimum = -INFINITY;
    c.maximum = INFINITY;
    c.closed = false;
    return c;
}

// Helper: Set cylinder transform and compute inverse
static inline void cylinder_set_transform(Cylinder* cylinder, Matrix4x4 transform) {
    cylinder->transform = transform;
    matrix_float4x4 mat = matrix_from_columns(transform.columns[0], transform.columns[1], 
                                               transform.columns[2], transform.columns[3]);
    matrix_float4x4 inv = matrix_invert(mat);
    cylinder->inverseTransform.columns[0] = inv.columns[0];
    cylinder->inverseTransform.columns[1] = inv.columns[1];
    cylinder->inverseTransform.columns[2] = inv.columns[2];
    cylinder->inverseTransform.columns[3] = inv.columns[3];
}

// Chapter 14: Group helper functions

// Helper: Create a group with identity transform
static inline Group group_create(int id) {
    Group g;
    g.id = id;
    g.transform = MATRIX4X4_IDENTITY;
    g.inverseTransform = MATRIX4X4_IDENTITY;
    g.child_count = 0;
    g.material = DEFAULT_MATERIAL;
    return g;
}

// Helper: Set group transform and compute inverse
static inline void group_set_transform(Group* group, Matrix4x4 transform) {
    group->transform = transform;
    matrix_float4x4 mat = matrix_from_columns(transform.columns[0], transform.columns[1], 
                                               transform.columns[2], transform.columns[3]);
    matrix_float4x4 inv = matrix_invert(mat);
    group->inverseTransform.columns[0] = inv.columns[0];
    group->inverseTransform.columns[1] = inv.columns[1];
    group->inverseTransform.columns[2] = inv.columns[2];
    group->inverseTransform.columns[3] = inv.columns[3];
}

// Helper: Add a child to a group
static inline int group_add_child(Group* group, int child_id, ShapeType child_type) {
    if (group->child_count >= MAX_GROUP_CHILDREN) {
        return 0;  // Failed - group is full
    }
    group->child_ids[group->child_count] = child_id;
    group->child_types[group->child_count] = child_type;
    group->child_count++;
    return 1;  // Success
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

// Chapter 11: Reflection and Refraction

// Helper: Reflect a vector around a normal
// in: incoming vector (typically points toward surface)
// normal: surface normal
// Returns reflected vector pointing away from surface
static inline vector_float4 reflect(vector_float4 in, vector_float4 normal) {
    return in - normal * 2.0f * simd_dot(in, normal);
}

// Predefined refractive indices
#define REFRACTIVE_INDEX_VACUUM 1.0f
#define REFRACTIVE_INDEX_AIR 1.0f
#define REFRACTIVE_INDEX_WATER 1.333f
#define REFRACTIVE_INDEX_GLASS 1.5f
#define REFRACTIVE_INDEX_DIAMOND 2.417f

// Helper: Compute refracted vector using Snell's law
// incident: incoming ray direction (normalized, pointing toward surface)
// normal: surface normal (pointing away from surface)
// n1: refractive index of source material
// n2: refractive index of destination material
// Returns refracted direction, or zero vector if total internal reflection occurs
static inline vector_float4 refract(vector_float4 incident, vector_float4 normal, float n1, float n2) {
    vector_float4 normalized_incident = simd_normalize(incident);
    vector_float4 normalized_normal = simd_normalize(normal);
    
    float cos_theta1 = -simd_dot(normalized_incident, normalized_normal);
    float sin_theta1_squared = 1.0f - cos_theta1 * cos_theta1;
    
    float n_ratio = n1 / n2;
    float sin_theta2_squared = n_ratio * n_ratio * sin_theta1_squared;
    
    // Check for total internal reflection
    if (sin_theta2_squared > 1.0f) {
        return (vector_float4){0, 0, 0, 0};  // Zero vector indicates TIR
    }
    
    float cos_theta2 = sqrtf(1.0f - sin_theta2_squared);
    
    // Refracted direction = (n1/n2) * incident + (n1/n2 * cos(theta1) - cos(theta2)) * normal
    vector_float4 result = normalized_incident * n_ratio + 
                          normalized_normal * (n_ratio * cos_theta1 - cos_theta2);
    result.w = 0.0f;  // Ensure it's a vector
    return result;
}

// Helper: Compute Fresnel effect (reflectance) using Schlick's approximation
// This determines how much light is reflected vs refracted
// cos_theta: cosine of angle between eye and normal
// refractive_index: ratio of n1/n2 (source/destination)
// Returns reflectance value (0-1), where 0 means all refracted, 1 means all reflected
static inline float schlick(float cos_theta, float refractive_index) {
    float r0 = powf((1.0f - refractive_index) / (1.0f + refractive_index), 2.0f);
    return r0 + (1.0f - r0) * powf(1.0f - cos_theta, 5.0f);
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

// Chapter 7: Making a Scene

// World constants
#define MAX_SPHERES_IN_WORLD 10
#define MAX_INTERSECTIONS_TOTAL 100

// World type: contains objects and lights
typedef struct {
    Sphere spheres[MAX_SPHERES_IN_WORLD];  // Objects in the world
    int sphere_count;                     // Number of spheres
    PointLight light;                     // Light source
} World;

// Helper: Create a default world with a light and two spheres
static inline World world_create_default(void) {
    World world;
    world.sphere_count = 2;
    
    // Light source
    world.light.position = (vector_float4){-10, 10, -10, 1};
    world.light.intensity = (vector_float4){1, 1, 1, 1};
    
    // First sphere: large, diffuse
    world.spheres[0] = sphere_create(1);
    world.spheres[0].transform = matrix_scaling(0.5, 0.5, 0.5);  // Small sphere
    matrix_float4x4 mat = matrix_from_columns(world.spheres[0].transform.columns[0],
                                              world.spheres[0].transform.columns[1],
                                              world.spheres[0].transform.columns[2],
                                              world.spheres[0].transform.columns[3]);
    matrix_float4x4 inv = matrix_invert(mat);
    world.spheres[0].inverseTransform.columns[0] = inv.columns[0];
    world.spheres[0].inverseTransform.columns[1] = inv.columns[1];
    world.spheres[0].inverseTransform.columns[2] = inv.columns[2];
    world.spheres[0].inverseTransform.columns[3] = inv.columns[3];
    
    // Second sphere: at origin
    world.spheres[1] = sphere_create(2);
    world.spheres[1].transform = matrix_translation(0, 0, 0);  // At origin
    matrix_float4x4 mat2 = matrix_from_columns(world.spheres[1].transform.columns[0],
                                                world.spheres[1].transform.columns[1],
                                                world.spheres[1].transform.columns[2],
                                                world.spheres[1].transform.columns[3]);
    matrix_float4x4 inv2 = matrix_invert(mat2);
    world.spheres[1].inverseTransform.columns[0] = inv2.columns[0];
    world.spheres[1].inverseTransform.columns[1] = inv2.columns[1];
    world.spheres[1].inverseTransform.columns[2] = inv2.columns[2];
    world.spheres[1].inverseTransform.columns[3] = inv2.columns[3];
    
    return world;
}

// Helper: Create an empty world
static inline World world_create_empty(void) {
    World world;
    world.sphere_count = 0;
    world.light = (PointLight){(vector_float4){0, 0, 0, 1}, (vector_float4){0, 0, 0, 1}};
    return world;
}

// Helper: Add sphere to world
static inline void world_add_sphere(World* world, Sphere sphere) {
    if (world->sphere_count < MAX_SPHERES_IN_WORLD) {
        world->spheres[world->sphere_count] = sphere;
        world->sphere_count++;
    }
}

// Helper: Find the hit from an array of intersections
// Returns the intersection with the smallest non-negative t value
static inline Intersection hit_from_array(Intersection* intersections, int count) {
    Intersection result;
    result.t = INFINITY;
    result.objectId = -1;
    
    for (int i = 0; i < count; i++) {
        if (intersections[i].t >= 0 && intersections[i].t < result.t) {
            result = intersections[i];
        }
    }
    
    if (result.objectId == -1) {
        result.t = -1;  // No hit found
    }
    
    return result;
}

// Helper: Ray-sphere intersection returning t values (simplified for world intersection)
static inline int intersect_sphere_simple(Sphere sphere, Ray ray, float* t0, float* t1) {
    // Transform ray to object space
    Ray object_ray = ray_transform(ray, sphere.inverseTransform);
    
    // Quadratic formula coefficients
    vector_float3 ray_origin = object_ray.origin.xyz;
    vector_float3 ray_direction = object_ray.direction.xyz;
    
    float a = simd_dot(ray_direction, ray_direction);
    float b = 2.0f * simd_dot(ray_origin, ray_direction);
    float c = simd_dot(ray_origin, ray_origin) - 1.0f;
    
    float discriminant = b*b - 4.0f*a*c;
    
    if (discriminant < 0) {
        return 0;  // No intersection
    }
    
    float sqrt_disc = sqrtf(discriminant);
    *t0 = (-b - sqrt_disc) / (2.0f * a);
    *t1 = (-b + sqrt_disc) / (2.0f * a);
    
    return (discriminant == 0) ? 1 : 2;
}

// Chapter 9: Plane intersection
// A plane is a flat surface extending infinitely in the x and z directions
// By default, it's at y=0 in object space

// Helper: Create a plane with identity transform
static inline Plane plane_create(int id) {
    Plane p;
    p.id = id;
    p.transform = MATRIX4X4_IDENTITY;
    p.inverseTransform = MATRIX4X4_IDENTITY;
    return p;
}

// Helper: Set plane transform
static inline void plane_set_transform(Plane* plane, Matrix4x4 transform) {
    plane->transform = transform;
    matrix_float4x4 mat = matrix_from_columns(transform.columns[0], transform.columns[1], 
                                              transform.columns[2], transform.columns[3]);
    matrix_float4x4 inv = matrix_invert(mat);
    plane->inverseTransform.columns[0] = inv.columns[0];
    plane->inverseTransform.columns[1] = inv.columns[1];
    plane->inverseTransform.columns[2] = inv.columns[2];
    plane->inverseTransform.columns[3] = inv.columns[3];
}

// Helper: Compute normal at a point on a plane
static inline vector_float4 plane_normal_at(Plane plane, vector_float4 point) {
    // Transform point to object space
    matrix_float4x4 inv = matrix_from_columns(plane.inverseTransform.columns[0],
                                               plane.inverseTransform.columns[1],
                                               plane.inverseTransform.columns[2],
                                               plane.inverseTransform.columns[3]);
    vector_float4 object_point = matrix_multiply(inv, point);
    
    // Normal in object space is (0, 1, 0, 0) - pointing up
    vector_float4 object_normal = {0.0f, 1.0f, 0.0f, 0.0f};
    
    // Transform normal back to world space
    matrix_float4x4 inv_transpose = matrix_transpose(inv);
    vector_float4 world_normal = matrix_multiply(inv_transpose, object_normal);
    world_normal.w = 0.0f;
    
    return simd_normalize(world_normal);
}

// Helper: Ray-plane intersection
// Returns 1 if hit, 0 if miss. Writes t value if hit.
// A plane at y=0 in object space: any point on plane has y=0
// Ray: origin + direction * t
// At intersection: origin.y + direction.y * t = 0
// Therefore: t = -origin.y / direction.y
static inline int intersect_plane(Plane plane, Ray ray, float* t) {
    // Transform ray to object space
    Ray object_ray = ray_transform(ray, plane.inverseTransform);
    
    // Check if ray is parallel to plane (direction.y is nearly 0)
    if (fabs(object_ray.direction.y) < 0.0001f) {
        return 0;  // Ray is parallel to plane, no intersection
    }
    
    // Compute intersection t
    float t_val = -object_ray.origin.y / object_ray.direction.y;
    
    // Only count intersections in front of the ray (t > 0)
    if (t_val > 0) {
        *t = t_val;
        return 1;  // Hit
    }
    
    return 0;  // Intersection is behind the ray
}

// Chapter 12: Ray-cube intersection
// A cube is bounded by 6 planes: x=±1, y=±1, z=±1
// Returns the number of intersections (0, 1, or 2) and writes t values
static inline int intersect_cube(Cube cube, Ray ray, float* t0_out, float* t1_out) {
    // Transform ray to cube's object space
    Ray object_ray = ray_transform(ray, cube.inverseTransform);
    
    vector_float4 origin = object_ray.origin;
    vector_float4 direction = object_ray.direction;
    
    float min_t = -INFINITY;
    float max_t = INFINITY;
    
    // Check each pair of parallel planes
    // X planes: x = -1 and x = 1
    if (fabs(direction.x) < 0.0001f) {
        // Ray is parallel to x planes, check if it's between them
        if (origin.x < -1.0f || origin.x > 1.0f) {
            return 0;  // Ray is outside the slab
        }
    } else {
        float tx0 = (-1.0f - origin.x) / direction.x;
        float tx1 = (1.0f - origin.x) / direction.x;
        if (tx0 > tx1) {
            float temp = tx0;
            tx0 = tx1;
            tx1 = temp;
        }
        if (tx0 > min_t) min_t = tx0;
        if (tx1 < max_t) max_t = tx1;
        if (min_t > max_t) return 0;
    }
    
    // Y planes: y = -1 and y = 1
    if (fabs(direction.y) < 0.0001f) {
        if (origin.y < -1.0f || origin.y > 1.0f) {
            return 0;
        }
    } else {
        float ty0 = (-1.0f - origin.y) / direction.y;
        float ty1 = (1.0f - origin.y) / direction.y;
        if (ty0 > ty1) {
            float temp = ty0;
            ty0 = ty1;
            ty1 = temp;
        }
        if (ty0 > min_t) min_t = ty0;
        if (ty1 < max_t) max_t = ty1;
        if (min_t > max_t) return 0;
    }
    
    // Z planes: z = -1 and z = 1
    if (fabs(direction.z) < 0.0001f) {
        if (origin.z < -1.0f || origin.z > 1.0f) {
            return 0;
        }
    } else {
        float tz0 = (-1.0f - origin.z) / direction.z;
        float tz1 = (1.0f - origin.z) / direction.z;
        if (tz0 > tz1) {
            float temp = tz0;
            tz0 = tz1;
            tz1 = temp;
        }
        if (tz0 > min_t) min_t = tz0;
        if (tz1 < max_t) max_t = tz1;
        if (min_t > max_t) return 0;
    }
    
    // We have valid intersection(s)
    // Check which ones are in front of the ray (t > 0)
    int count = 0;
    
    if (min_t > 0) {
        *t0_out = min_t;
        count = 1;
    }
    
    if (max_t > 0 && max_t != min_t) {
        if (count == 0) {
            *t0_out = max_t;
            count = 1;
        } else {
            *t1_out = max_t;
            count = 2;
        }
    }
    
    return count;
}

// Chapter 12: Compute normal at a point on a cube
// The normal points outward from the cube face that the point is on
static inline vector_float4 cube_normal_at(Cube cube, vector_float4 point) {
    // Transform point to cube's object space
    matrix_float4x4 inv = matrix_from_columns(cube.inverseTransform.columns[0],
                                               cube.inverseTransform.columns[1],
                                               cube.inverseTransform.columns[2],
                                               cube.inverseTransform.columns[3]);
    vector_float4 object_point = matrix_multiply(inv, point);
    
    // Find the largest component to determine which face we're on
    float abs_x = fabs(object_point.x);
    float abs_y = fabs(object_point.y);
    float abs_z = fabs(object_point.z);
    
    vector_float4 object_normal;
    
    if (abs_x >= abs_y && abs_x >= abs_z) {
        // On an x face
        object_normal = (vector_float4){object_point.x > 0 ? 1.0f : -1.0f, 0, 0, 0};
    } else if (abs_y >= abs_x && abs_y >= abs_z) {
        // On a y face
        object_normal = (vector_float4){0, object_point.y > 0 ? 1.0f : -1.0f, 0, 0};
    } else {
        // On a z face
        object_normal = (vector_float4){0, 0, object_point.z > 0 ? 1.0f : -1.0f, 0};
    }
    
    // Transform normal back to world space
    // For normals, we use the transpose of the inverse
    matrix_float4x4 inv_transpose = matrix_transpose(inv);
    vector_float4 world_normal = matrix_multiply(inv_transpose, object_normal);
    world_normal.w = 0.0f;
    
    return simd_normalize(world_normal);
}

// Chapter 13: Ray-cylinder intersection
// A cylinder in object space is defined by x² + z² = 1 (unit radius, infinite along y)
// Returns the number of intersections (0, 1, or 2) and writes t values
static inline int intersect_cylinder(Cylinder cylinder, Ray ray, float* t0_out, float* t1_out) {
    // Transform ray to cylinder's object space
    Ray object_ray = ray_transform(ray, cylinder.inverseTransform);
    
    vector_float4 origin = object_ray.origin;
    vector_float4 direction = object_ray.direction;
    
    // Cylinder equation: x² + z² = 1
    // Substituting ray: (ox + t*dx)² + (oz + t*dz)² = 1
    // Expanding: (dx² + dz²)t² + 2(ox*dx + oz*dz)t + (ox² + oz² - 1) = 0
    
    float a = direction.x * direction.x + direction.z * direction.z;
    
    // If a is nearly 0, the ray is parallel to the y-axis (cylinder axis)
    // This means the ray won't hit the cylinder walls
    if (fabs(a) < 0.0001f) {
        // Ray is parallel to cylinder axis, check if it hits the caps
        if (cylinder.closed) {
            // Check cap intersections
            // This is simplified - full implementation would check both caps
        }
        return 0;
    }
    
    float b = 2.0f * (origin.x * direction.x + origin.z * direction.z);
    float c = origin.x * origin.x + origin.z * origin.z - 1.0f;
    
    // Calculate discriminant
    float discriminant = b * b - 4.0f * a * c;
    
    if (discriminant < 0.0f) {
        return 0;  // No real solutions, ray misses cylinder
    }
    
    // Calculate t values
    float sqrt_disc = sqrtf(discriminant);
    float t0 = (-b - sqrt_disc) / (2.0f * a);
    float t1 = (-b + sqrt_disc) / (2.0f * a);
    
    // Check y bounds for truncation
    int count = 0;
    
    // Check t0
    if (t0 >= 0.001f) {
        float y0 = origin.y + t0 * direction.y;
        if (y0 >= cylinder.minimum && y0 <= cylinder.maximum) {
            *t0_out = t0;
            count = 1;
        }
    }
    
    // Check t1
    if (t1 >= 0.001f) {
        float y1 = origin.y + t1 * direction.y;
        if (y1 >= cylinder.minimum && y1 <= cylinder.maximum) {
            if (count == 0) {
                *t0_out = t1;
                count = 1;
            } else {
                *t1_out = t1;
                count = 2;
            }
        }
    }
    
    // Check caps if cylinder is closed
    if (cylinder.closed && (count < 2 || t1 < 0.001f)) {
        // Check intersection with caps (simplified - assumes ray can hit caps)
        if (fabs(direction.y) > 0.0001f) {
            // Check bottom cap at y = minimum
            float t_cap0 = (cylinder.minimum - origin.y) / direction.y;
            if (t_cap0 > 0.001f) {
                float x = origin.x + t_cap0 * direction.x;
                float z = origin.z + t_cap0 * direction.z;
                if (x * x + z * z <= 1.0f) {
                    if (count == 0) {
                        *t0_out = t_cap0;
                        count = 1;
                    } else if (count == 1) {
                        *t1_out = t_cap0;
                        count = 2;
                    }
                }
            }
            
            // Check top cap at y = maximum
            float t_cap1 = (cylinder.maximum - origin.y) / direction.y;
            if (t_cap1 > 0.001f) {
                float x = origin.x + t_cap1 * direction.x;
                float z = origin.z + t_cap1 * direction.z;
                if (x * x + z * z <= 1.0f) {
                    if (count == 0) {
                        *t0_out = t_cap1;
                        count = 1;
                    } else if (count == 1 && t_cap1 != *t0_out) {
                        *t1_out = t_cap1;
                        count = 2;
                    }
                }
            }
        }
    }
    
    return count;
}

// Chapter 13: Compute normal at a point on a cylinder
static inline vector_float4 cylinder_normal_at(Cylinder cylinder, vector_float4 point) {
    // Transform point to cylinder's object space
    matrix_float4x4 inv = matrix_from_columns(cylinder.inverseTransform.columns[0],
                                               cylinder.inverseTransform.columns[1],
                                               cylinder.inverseTransform.columns[2],
                                               cylinder.inverseTransform.columns[3]);
    vector_float4 object_point = matrix_multiply(inv, point);
    
    vector_float4 object_normal;
    
    // Calculate distance from y-axis in object space
    float dist_sq = object_point.x * object_point.x + object_point.z * object_point.z;
    
    // Check if we're on the top cap
    if (object_point.y >= cylinder.maximum - 0.0001f) {
        object_normal = (vector_float4){0, 1, 0, 0};
    }
    // Check if we're on the bottom cap
    else if (object_point.y <= cylinder.minimum + 0.0001f) {
        object_normal = (vector_float4){0, -1, 0, 0};
    }
    // Otherwise we're on the side
    else {
        object_normal = (vector_float4){object_point.x, 0, object_point.z, 0};
    }
    
    // Transform normal back to world space
    matrix_float4x4 inv_transpose = matrix_transpose(inv);
    vector_float4 world_normal = matrix_multiply(inv_transpose, object_normal);
    world_normal.w = 0.0f;
    
    return simd_normalize(world_normal);
}

// Chapter 14: Ray-group intersection
// Note: Full implementation would require passing World to look up child shapes
// For now, we provide a simplified interface

// Helper: Intersect ray with world, return all intersections
// intersections array must be large enough (MAX_INTERSECTIONS_TOTAL)
static inline int intersect_world(World world, Ray ray, Intersection* intersections) {
    int count = 0;
    
    for (int i = 0; i < world.sphere_count; i++) {
        float t0, t1;
        int num_hits = intersect_sphere_simple(world.spheres[i], ray, &t0, &t1);
        
        if (num_hits >= 1 && count < MAX_INTERSECTIONS_TOTAL) {
            intersections[count].t = t0;
            intersections[count].objectId = world.spheres[i].id;
            count++;
        }
        
        if (num_hits >= 2 && count < MAX_INTERSECTIONS_TOTAL) {
            intersections[count].t = t1;
            intersections[count].objectId = world.spheres[i].id;
            count++;
        }
    }
    
    // Sort intersections by t value (bubble sort)
    for (int i = 0; i < count - 1; i++) {
        for (int j = 0; j < count - i - 1; j++) {
            if (intersections[j].t > intersections[j + 1].t) {
                Intersection temp = intersections[j];
                intersections[j] = intersections[j + 1];
                intersections[j + 1] = temp;
            }
        }
    }
    
    return count;
}

// Chapter 8: Shadows - Forward declarations
static inline int is_shadowed(World world, vector_float4 point);
static inline vector_float4 lighting_with_shadow(Material material, PointLight light, 
                                                vector_float4 point, vector_float4 eye_vector, 
                                                vector_float4 normal, int in_shadow);

// Helper: Shade a hit point
// This combines lighting with the material of the hit object
static inline vector_float4 shade_hit(World world, Intersection hit, Ray ray) {
    // Find the sphere that was hit
    Sphere hit_sphere;
    int found = 0;
    for (int i = 0; i < world.sphere_count; i++) {
        if (world.spheres[i].id == hit.objectId) {
            hit_sphere = world.spheres[i];
            found = 1;
            break;
        }
    }
    
    if (!found) {
        return (vector_float4){0, 0, 0, 1};  // Black if object not found
    }
    
    // Compute hit point
    vector_float4 point = ray_position(ray, hit.t);
    
    // Compute normal at hit point
    vector_float4 normal = sphere_normal_at(hit_sphere, point);
    
    // Eye vector is negative of ray direction
    vector_float4 eye = -ray.direction;
    
    // Check if point is in shadow
    int in_shadow = is_shadowed(world, point);
    
    // Use the sphere's material
    Material material = hit_sphere.material;
    
    // Compute lighting with shadow
    return lighting_with_shadow(material, world.light, point, eye, normal, in_shadow);
}

// Helper: Compute color at a point in the world (main rendering function)
static inline vector_float4 color_at(World world, Ray ray) {
    Intersection intersections[MAX_INTERSECTIONS_TOTAL];
    int count = intersect_world(world, ray, intersections);
    
    if (count == 0) {
        return (vector_float4){0, 0, 0, 1};  // Black background
    }
    
    Intersection hit = hit_from_array(intersections, count);
    
    if (hit.objectId == -1) {
        return (vector_float4){0, 0, 0, 1};  // No hit
    }
    
    return shade_hit(world, hit, ray);
}

// Chapter 11: Reflection and Refraction - Part 2

// Maximum recursion depth for reflection/refraction
#define MAX_RECURSION_DEPTH 5

// Forward declarations for recursive shading
static inline vector_float4 color_at_recursive(World world, Ray ray, int remaining);
static inline vector_float4 reflected_color(World world, Intersection hit, Ray ray, int remaining);
static inline vector_float4 refracted_color(World world, Intersection hit, Ray ray, int remaining);
static inline vector_float4 shade_hit_recursive(World world, Intersection hit, Ray ray, int remaining);

// Helper: Compute reflected color by casting a reflection ray
// remaining: recursion depth remaining
static inline vector_float4 reflected_color(World world, Intersection hit, Ray ray, int remaining) {
    // Find the sphere that was hit first
    Sphere hit_sphere;
    int found = 0;
    for (int i = 0; i < world.sphere_count; i++) {
        if (world.spheres[i].id == hit.objectId) {
            hit_sphere = world.spheres[i];
            found = 1;
            break;
        }
    }
    
    if (!found) {
        return (vector_float4){0, 0, 0, 1};
    }
    
    // Check if material is reflective and we haven't hit recursion limit
    Material material = hit_sphere.material;
    if (remaining <= 0 || material.reflective == 0.0f) {
        return (vector_float4){0, 0, 0, 1};
    }
    
    // Compute hit point and normal
    vector_float4 point = ray_position(ray, hit.t);
    vector_float4 normal = sphere_normal_at(hit_sphere, point);
    
    // Compute reflected direction
    vector_float4 reflect_dir = reflect(ray.direction, normal);
    reflect_dir.w = 0.0f;
    reflect_dir = simd_normalize(reflect_dir);
    
    // Create reflection ray, offset slightly to avoid self-intersection
    Ray reflect_ray;
    reflect_ray.origin = point + normal * 0.001f;
    reflect_ray.direction = reflect_dir;
    
    // Recursively trace the reflection ray
    vector_float4 reflected = color_at_recursive(world, reflect_ray, remaining - 1);
    
    // Scale by material's reflectivity
    return reflected * material.reflective;
}

// Helper: Compute refracted color by casting a refraction ray
// remaining: recursion depth remaining  
static inline vector_float4 refracted_color(World world, Intersection hit, Ray ray, int remaining) {
    // Find the sphere that was hit first
    Sphere hit_sphere;
    int found = 0;
    for (int i = 0; i < world.sphere_count; i++) {
        if (world.spheres[i].id == hit.objectId) {
            hit_sphere = world.spheres[i];
            found = 1;
            break;
        }
    }
    
    if (!found) {
        return (vector_float4){0, 0, 0, 1};
    }
    
    // Check if material is transparent and we haven't hit recursion limit
    Material material = hit_sphere.material;
    if (remaining <= 0 || material.transparency == 0.0f) {
        return (vector_float4){0, 0, 0, 1};
    }
    
    // Compute hit point and normal
    vector_float4 point = ray_position(ray, hit.t);
    vector_float4 normal = sphere_normal_at(hit_sphere, point);
    
    // Check if we're entering or exiting the object
    // If ray direction and normal are on same side, we're exiting
    float n1, n2;
    if (simd_dot(ray.direction, normal) > 0) {
        // Exiting: inside to outside
        n1 = material.refractive_index;
        n2 = 1.0f;  // Assuming air
        normal = -normal;  // Flip normal
    } else {
        // Entering: outside to inside
        n1 = 1.0f;  // Assuming air
        n2 = material.refractive_index;
    }
    
    // Compute refracted direction using Snell's law
    vector_float4 refract_dir = refract(ray.direction, normal, n1, n2);
    
    // Check for total internal reflection
    float dir_len = simd_length(refract_dir.xyz);
    if (dir_len < 0.0001f) {
        return (vector_float4){0, 0, 0, 1};  // Total internal reflection
    }
    
    refract_dir = simd_normalize(refract_dir);
    
    // Create refraction ray, offset in direction of refraction to avoid self-intersection
    Ray refract_ray;
    refract_ray.origin = point - normal * 0.001f;  // Offset opposite to normal
    refract_ray.direction = refract_dir;
    
    // Recursively trace the refraction ray
    vector_float4 refracted = color_at_recursive(world, refract_ray, remaining - 1);
    
    // Scale by material's transparency
    return refracted * material.transparency;
}

// Helper: Shade a hit point with recursive reflection and refraction
// This combines base lighting with reflected and refracted colors
static inline vector_float4 shade_hit_recursive(World world, Intersection hit, Ray ray, int remaining) {
    // Get base color (surface lighting)
    vector_float4 surface = shade_hit(world, hit, ray);
    
    // Get reflected color
    vector_float4 reflected = reflected_color(world, hit, ray, remaining);
    
    // Get refracted color
    vector_float4 refracted = refracted_color(world, hit, ray, remaining);
    
    // Combine: surface + reflected + refracted
    vector_float4 result = surface + reflected + refracted;
    result.w = 1.0f;
    
    return result;
}

// Helper: Compute color at a point in the world with recursion
// remaining: recursion depth remaining (starts at MAX_RECURSION_DEPTH)
static inline vector_float4 color_at_recursive(World world, Ray ray, int remaining) {
    if (remaining <= 0) {
        return (vector_float4){0, 0, 0, 1};  // Black at recursion limit
    }
    
    Intersection intersections[MAX_INTERSECTIONS_TOTAL];
    int count = intersect_world(world, ray, intersections);
    
    if (count == 0) {
        return (vector_float4){0, 0, 0, 1};  // Black background
    }
    
    Intersection hit = hit_from_array(intersections, count);
    
    if (hit.objectId == -1) {
        return (vector_float4){0, 0, 0, 1};  // No hit
    }
    
    return shade_hit_recursive(world, hit, ray, remaining);
}

// Chapter 8: Shadows

// Helper: Check if a point is in shadow
// Casts a ray from point toward light, returns true if anything blocks it
static inline int is_shadowed(World world, vector_float4 point) {
    // Vector from point to light
    vector_float4 light_dir = world.light.position - point;
    float distance = simd_length(light_dir.xyz);
    light_dir = simd_normalize(light_dir);
    
    // Create shadow ray starting slightly offset from point to avoid self-intersection
    Ray shadow_ray;
    shadow_ray.origin = point + light_dir * 0.001f;  // Small offset
    shadow_ray.direction = light_dir;
    
    // Check for intersections
    Intersection intersections[MAX_INTERSECTIONS_TOTAL];
    int count = intersect_world(world, shadow_ray, intersections);
    
    // Check if any intersection is between point and light
    for (int i = 0; i < count; i++) {
        if (intersections[i].t > 0 && intersections[i].t < distance) {
            return 1;  // In shadow
        }
    }
    
    return 0;  // Not in shadow
}

// Helper: Compute lighting with shadow support
static inline vector_float4 lighting_with_shadow(Material material, PointLight light, 
                                                vector_float4 point, vector_float4 eye_vector, 
                                                vector_float4 normal, int in_shadow) {
    // Combine material color with light intensity
    vector_float4 effective_color = material.color * light.intensity;
    effective_color.w = 1.0;
    
    // Find direction to light
    vector_float4 light_vector = simd_normalize(light.position - point);
    
    // Compute ambient contribution (always present, even in shadow)
    vector_float4 ambient = effective_color * material.ambient;
    ambient.w = 1.0;
    
    // If in shadow, only return ambient light
    if (in_shadow) {
        vector_float4 result = ambient;
        result.w = 1.0;
        return result;
    }
    
    // light_dot_normal represents cosine of angle between light vector and normal
    float light_dot_normal = simd_dot(light_vector, normal);
    
    vector_float4 diffuse = {0, 0, 0, 0};
    vector_float4 specular = {0, 0, 0, 0};
    
    if (light_dot_normal >= 0) {
        // Compute diffuse contribution
        diffuse = effective_color * material.diffuse * light_dot_normal;
        diffuse.w = 1.0;
        
        // Compute reflection vector
        vector_float4 reflect_vector = -light_vector + normal * 2.0f * light_dot_normal;
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
    result.w = 1.0;
    return result;
}

// Chapter 7: Camera and View Transform

// Camera type for rendering
typedef struct {
    int hsize;          // Horizontal size (in pixels)
    int vsize;          // Vertical size (in pixels)
    float field_of_view; // Field of view angle in radians
    Matrix4x4 transform; // Camera transformation matrix
    float pixel_size;   // Precomputed pixel size
    float half_width;   // Precomputed half width
    float half_height;  // Precomputed half height
} Camera;

// Helper: Create a view transformation matrix
// Transforms world space to camera space
// from: camera position, to: point to look at, up: up vector
static inline Matrix4x4 view_transform(vector_float4 from, vector_float4 to, vector_float4 up) {
    // Compute forward vector (pointing from camera to target)
    vector_float4 forward = simd_normalize(to - from);
    forward.w = 0;
    
    // Compute left vector (cross product of forward and normalized up)
    vector_float4 upn = simd_normalize(up);
    upn.w = 0;
    vector_float3 left3 = simd_cross(forward.xyz, upn.xyz);
    vector_float4 left = {left3.x, left3.y, left3.z, 0};
    left = simd_normalize(left);
    
    // Compute true up vector (cross product of left and forward)
    vector_float3 true_up3 = simd_cross(left.xyz, forward.xyz);
    vector_float4 true_up = {true_up3.x, true_up3.y, true_up3.z, 0};
    
    // Create orientation matrix
    Matrix4x4 orientation;
    orientation.columns[0] = (vector_float4){left.x, true_up.x, -forward.x, 0};
    orientation.columns[1] = (vector_float4){left.y, true_up.y, -forward.y, 0};
    orientation.columns[2] = (vector_float4){left.z, true_up.z, -forward.z, 0};
    orientation.columns[3] = (vector_float4){0, 0, 0, 1};
    
    // Create translation matrix
    Matrix4x4 translation = matrix_translation(-from.x, -from.y, -from.z);
    
    // Combine: orientation * translation
    matrix_float4x4 o = matrix_from_columns(orientation.columns[0], orientation.columns[1],
                                           orientation.columns[2], orientation.columns[3]);
    matrix_float4x4 t = matrix_from_columns(translation.columns[0], translation.columns[1],
                                           translation.columns[2], translation.columns[3]);
    matrix_float4x4 result = matrix_multiply(o, t);
    
    Matrix4x4 view;
    view.columns[0] = result.columns[0];
    view.columns[1] = result.columns[1];
    view.columns[2] = result.columns[2];
    view.columns[3] = result.columns[3];
    return view;
}

// Helper: Create a camera
static inline Camera camera_create(int hsize, int vsize, float field_of_view) {
    Camera cam;
    cam.hsize = hsize;
    cam.vsize = vsize;
    cam.field_of_view = field_of_view;
    cam.transform = MATRIX4X4_IDENTITY;
    
    // Precompute pixel size
    float half_view = tanf(field_of_view / 2.0f);
    float aspect = (float)hsize / (float)vsize;
    
    if (aspect >= 1.0f) {
        cam.half_width = half_view;
        cam.half_height = half_view / aspect;
    } else {
        cam.half_width = half_view * aspect;
        cam.half_height = half_view;
    }
    
    cam.pixel_size = (cam.half_width * 2.0f) / (float)hsize;
    
    return cam;
}

// Helper: Ray for pixel
// Computes the ray that originates from the camera and passes through the given pixel
static inline Ray ray_for_pixel(Camera cam, int px, int py) {
    // Offset from edge of canvas to center of pixel
    float xoffset = (px + 0.5f) * cam.pixel_size;
    float yoffset = (py + 0.5f) * cam.pixel_size;
    
    // Untransformed coordinates
    float world_x = cam.half_width - xoffset;
    float world_y = cam.half_height - yoffset;
    
    // Transform canvas point and origin using camera transform
    vector_float4 pixel = (vector_float4){world_x, world_y, -1, 1};
    vector_float4 origin = (vector_float4){0, 0, 0, 1};
    
    matrix_float4x4 inv_transform = matrix_invert(matrix_from_columns(cam.transform.columns[0],
                                                                       cam.transform.columns[1],
                                                                       cam.transform.columns[2],
                                                                       cam.transform.columns[3]));
    
    vector_float4 pixel_world = matrix_multiply(inv_transform, pixel);
    vector_float4 origin_world = matrix_multiply(inv_transform, origin);
    
    vector_float4 direction = simd_normalize(pixel_world - origin_world);
    
    Ray ray;
    ray.origin = origin_world;
    ray.direction = direction;
    return ray;
}

// Helper: Create a world with the Chapter 7 scene (6 spheres)
static inline World world_create_scene(void) {
    World world;
    world.sphere_count = 0;
    
    // Light source - white, from above and left
    world.light.position = (vector_float4){-10, 10, -10, 1};
    world.light.intensity = (vector_float4){1, 1, 1, 1};
    
    // 1. The floor - extremely flattened sphere with matte texture
    Sphere floor = sphere_create(1);
    floor.transform = matrix_scaling(10, 0.01, 10);
    matrix_float4x4 floor_mat = matrix_from_columns(floor.transform.columns[0],
                                                    floor.transform.columns[1],
                                                    floor.transform.columns[2],
                                                    floor.transform.columns[3]);
    matrix_float4x4 floor_inv = matrix_invert(floor_mat);
    floor.inverseTransform.columns[0] = floor_inv.columns[0];
    floor.inverseTransform.columns[1] = floor_inv.columns[1];
    floor.inverseTransform.columns[2] = floor_inv.columns[2];
    floor.inverseTransform.columns[3] = floor_inv.columns[3];
    
    // Create floor material (matte, light color, no specular)
    Material floor_material = material_create((vector_float4){1, 0.9, 0.9, 1}, 0.1f, 0.9f, 0.0f, 200.0f);
    // Note: Material is embedded in shading logic, we'll handle this in the shader
    world_add_sphere(&world, floor);
    
    // 2. Left wall - same scale/color as floor, rotated and translated
    Sphere left_wall = sphere_create(2);
    Matrix4x4 left_scale = matrix_scaling(10, 0.01, 10);
    Matrix4x4 left_rot_x = matrix_rotation_x(M_PI_2);
    Matrix4x4 left_rot_y = matrix_rotation_y(-M_PI_4);
    Matrix4x4 left_trans = matrix_translation(0, 0, 5);
    
    // Transform order: scale, rot_x, rot_y, translate
    Matrix4x4 left_temp = left_scale;
    matrix_float4x4 left_m = matrix_from_columns(left_temp.columns[0], left_temp.columns[1],
                                                  left_temp.columns[2], left_temp.columns[3]);
    matrix_float4x4 left_m2 = matrix_from_columns(left_rot_x.columns[0], left_rot_x.columns[1],
                                                   left_rot_x.columns[2], left_rot_x.columns[3]);
    matrix_float4x4 left_m3 = matrix_from_columns(left_rot_y.columns[0], left_rot_y.columns[1],
                                                   left_rot_y.columns[2], left_rot_y.columns[3]);
    matrix_float4x4 left_m4 = matrix_from_columns(left_trans.columns[0], left_trans.columns[1],
                                                   left_trans.columns[2], left_trans.columns[3]);
    
    matrix_float4x4 left_result = matrix_multiply(left_m2, left_m);
    left_result = matrix_multiply(left_m3, left_result);
    left_result = matrix_multiply(left_m4, left_result);
    
    left_wall.transform.columns[0] = left_result.columns[0];
    left_wall.transform.columns[1] = left_result.columns[1];
    left_wall.transform.columns[2] = left_result.columns[2];
    left_wall.transform.columns[3] = left_result.columns[3];
    left_wall.inverseTransform.columns[0] = matrix_invert(left_result).columns[0];
    left_wall.inverseTransform.columns[1] = matrix_invert(left_result).columns[1];
    left_wall.inverseTransform.columns[2] = matrix_invert(left_result).columns[2];
    left_wall.inverseTransform.columns[3] = matrix_invert(left_result).columns[3];
    
    world_add_sphere(&world, left_wall);
    
    // 3. Right wall - identical but rotated opposite in Y
    Sphere right_wall = sphere_create(3);
    Matrix4x4 right_rot_y = matrix_rotation_y(M_PI_4);
    
    matrix_float4x4 right_m3 = matrix_from_columns(right_rot_y.columns[0], right_rot_y.columns[1],
                                                     right_rot_y.columns[2], right_rot_y.columns[3]);
    
    matrix_float4x4 right_result = matrix_multiply(left_m2, left_m);  // Same scale and rot_x as left
    right_result = matrix_multiply(right_m3, right_result);
    right_result = matrix_multiply(left_m4, right_result);
    
    right_wall.transform.columns[0] = right_result.columns[0];
    right_wall.transform.columns[1] = right_result.columns[1];
    right_wall.transform.columns[2] = right_result.columns[2];
    right_wall.transform.columns[3] = right_result.columns[3];
    right_wall.inverseTransform.columns[0] = matrix_invert(right_result).columns[0];
    right_wall.inverseTransform.columns[1] = matrix_invert(right_result).columns[1];
    right_wall.inverseTransform.columns[2] = matrix_invert(right_result).columns[2];
    right_wall.inverseTransform.columns[3] = matrix_invert(right_result).columns[3];
    
    world_add_sphere(&world, right_wall);
    
    // 4. Middle sphere - unit sphere, translated, green
    Sphere middle = sphere_create(4);
    middle.transform = matrix_translation(-0.5, 1, 0.5);
    matrix_float4x4 middle_mat = matrix_from_columns(middle.transform.columns[0],
                                                    middle.transform.columns[1],
                                                    middle.transform.columns[2],
                                                    middle.transform.columns[3]);
    matrix_float4x4 middle_inv = matrix_invert(middle_mat);
    middle.inverseTransform.columns[0] = middle_inv.columns[0];
    middle.inverseTransform.columns[1] = middle_inv.columns[1];
    middle.inverseTransform.columns[2] = middle_inv.columns[2];
    middle.inverseTransform.columns[3] = middle_inv.columns[3];
    
    world_add_sphere(&world, middle);
    
    // 5. Right sphere - smaller, green, scaled in half
    Sphere right = sphere_create(5);
    Matrix4x4 right_s_trans = matrix_translation(1.5, 0.5, -0.5);
    Matrix4x4 right_s_scale = matrix_scaling(0.5, 0.5, 0.5);
    
    matrix_float4x4 right_s_t = matrix_from_columns(right_s_trans.columns[0], right_s_trans.columns[1],
                                                     right_s_trans.columns[2], right_s_trans.columns[3]);
    matrix_float4x4 right_s_s = matrix_from_columns(right_s_scale.columns[0], right_s_scale.columns[1],
                                                     right_s_scale.columns[2], right_s_scale.columns[3]);
    matrix_float4x4 right_s_result = matrix_multiply(right_s_t, right_s_s);
    
    right.transform.columns[0] = right_s_result.columns[0];
    right.transform.columns[1] = right_s_result.columns[1];
    right.transform.columns[2] = right_s_result.columns[2];
    right.transform.columns[3] = right_s_result.columns[3];
    right.inverseTransform.columns[0] = matrix_invert(right_s_result).columns[0];
    right.inverseTransform.columns[1] = matrix_invert(right_s_result).columns[1];
    right.inverseTransform.columns[2] = matrix_invert(right_s_result).columns[2];
    right.inverseTransform.columns[3] = matrix_invert(right_s_result).columns[3];
    
    world_add_sphere(&world, right);
    
    // 6. Left sphere - smallest, yellow, scaled by third
    Sphere left = sphere_create(6);
    Matrix4x4 left_s_trans = matrix_translation(-1.5, 0.33, -0.75);
    Matrix4x4 left_s_scale = matrix_scaling(0.33, 0.33, 0.33);
    
    matrix_float4x4 left_s_t = matrix_from_columns(left_s_trans.columns[0], left_s_trans.columns[1],
                                                    left_s_trans.columns[2], left_s_trans.columns[3]);
    matrix_float4x4 left_s_s = matrix_from_columns(left_s_scale.columns[0], left_s_scale.columns[1],
                                                    left_s_scale.columns[2], left_s_scale.columns[3]);
    matrix_float4x4 left_s_result = matrix_multiply(left_s_t, left_s_s);
    
    left.transform.columns[0] = left_s_result.columns[0];
    left.transform.columns[1] = left_s_result.columns[1];
    left.transform.columns[2] = left_s_result.columns[2];
    left.transform.columns[3] = left_s_result.columns[3];
    left.inverseTransform.columns[0] = matrix_invert(left_s_result).columns[0];
    left.inverseTransform.columns[1] = matrix_invert(left_s_result).columns[1];
    left.inverseTransform.columns[2] = matrix_invert(left_s_result).columns[2];
    left.inverseTransform.columns[3] = matrix_invert(left_s_result).columns[3];
    
    world_add_sphere(&world, left);
    
    return world;
}

// Helper: Create default camera for Chapter 7 scene
static inline Camera camera_create_scene(int hsize, int vsize) {
    Camera cam = camera_create(hsize, vsize, M_PI / 3.0f);
    
    vector_float4 from = (vector_float4){0, 1.5, -5, 1};
    vector_float4 to = (vector_float4){0, 1, 0, 1};
    vector_float4 up = (vector_float4){0, 1, 0, 0};
    
    cam.transform = view_transform(from, to, up);
    
    return cam;
}

#endif /* !METAL_AVAILABLE */

#endif /* SharedTypes_h */
