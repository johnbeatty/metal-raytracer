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
    
    // Default material (white)
    Material material = DEFAULT_MATERIAL;
    
    // Compute lighting
    return lighting(material, world.light, point, eye, normal);
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
