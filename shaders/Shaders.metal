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
fragment float4 fragmentShader(RasterizerData in [[stage_in]],
                               texture2d<float> sphereTexture [[ texture(0) ]])
{
    constexpr sampler textureSampler (mag_filter::linear, min_filter::linear);
    
    // Sample from the sphere texture using texture coordinates
    float4 color = sphereTexture.sample(textureSampler, in.textureCoordinate);
    
    return color;
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


// ---------------------------
// Chapter 5: Ray-Sphere Intersections (Compute)
// ---------------------------

// Helper: Transform a ray by a matrix
Ray transform_ray(Ray ray, float4x4 matrix)
{
    Ray result;
    result.origin = matrix * ray.origin;
    result.direction = matrix * ray.direction;
    return result;
}

// Helper: Intersect a ray with a unit sphere at origin
// Returns number of intersections (0, 1, or 2) and writes t values to out_t0 and out_t1
int intersect_unit_sphere(Ray ray, thread float* out_t0, thread float* out_t1)
{
    // Sphere equation: x^2 + y^2 + z^2 = 1
    // Ray: origin + direction * t
    // Substituting: (origin + direction*t)^2 = 1
    // Expanding: direction^2 * t^2 + 2*origin*direction*t + origin^2 - 1 = 0
    // This is a quadratic: a*t^2 + b*t + c = 0
    
    float3 ray_origin = ray.origin.xyz;
    float3 ray_direction = ray.direction.xyz;
    
    float a = dot(ray_direction, ray_direction);
    float b = 2.0 * dot(ray_origin, ray_direction);
    float c = dot(ray_origin, ray_origin) - 1.0;
    
    float discriminant = b*b - 4.0*a*c;
    
    if (discriminant < 0.0) {
        return 0;  // No intersection
    }
    
    float sqrt_disc = sqrt(discriminant);
    *out_t0 = (-b - sqrt_disc) / (2.0 * a);
    *out_t1 = (-b + sqrt_disc) / (2.0 * a);
    
    if (discriminant == 0.0) {
        return 1;  // Tangent (one intersection)
    }
    
    return 2;  // Two intersections
}

// Kernel: Intersect rays with spheres
// Each thread processes one ray-sphere pair
// Output: up to 2 intersections per ray written to intersection buffer
kernel void ray_sphere_intersect(device const Ray* rays [[ buffer(0) ]],
                                 device const Sphere* spheres [[ buffer(1) ]],
                                 device Intersection* intersections [[ buffer(2) ]],
                                 device int* intersection_counts [[ buffer(3) ]],
                                 uint ray_index [[ thread_position_in_grid ]])
{
    Ray ray = rays[ray_index];
    Sphere sphere = spheres[0];  // For now, assume one sphere
    
    // Transform ray to object space using sphere's inverse transform
    float4x4 inverse_transform = float4x4(sphere.inverseTransform.columns[0],
                                          sphere.inverseTransform.columns[1],
                                          sphere.inverseTransform.columns[2],
                                          sphere.inverseTransform.columns[3]);
    Ray object_ray = transform_ray(ray, inverse_transform);
    
    // Intersect with unit sphere
    float t0, t1;
    int count = intersect_unit_sphere(object_ray, &t0, &t1);
    
    // Write intersection count
    intersection_counts[ray_index] = count;
    
    // Write intersections (base index for this ray is ray_index * 2)
    int base_index = ray_index * 2;
    if (count >= 1) {
        intersections[base_index].t = t0;
        intersections[base_index].objectId = sphere.id;
    }
    if (count >= 2) {
        intersections[base_index + 1].t = t1;
        intersections[base_index + 1].objectId = sphere.id;
    }
}

// ---------------------------
// Chapter 5: Render sphere silhouette to texture
// ---------------------------

// ---------------------------
// Chapter 6: Helper functions for 3D rendering with lighting
// ---------------------------

// Helper: Compute sphere normal at a point (Metal version)
float4 sphere_normal_at_metal(Sphere sphere, float4 point)
{
    // Transform point to object space
    float4x4 inv = float4x4(sphere.inverseTransform.columns[0],
                           sphere.inverseTransform.columns[1],
                           sphere.inverseTransform.columns[2],
                           sphere.inverseTransform.columns[3]);
    float4 object_point = inv * point;
    
    // Compute normal in object space
    float4 object_normal = object_point;
    object_normal.w = 0.0;
    
    // Transform normal back to world space using inverse transpose
    float4x4 inv_transpose = transpose(inv);
    float4 world_normal = inv_transpose * object_normal;
    world_normal.w = 0.0;
    
    return normalize(world_normal);
}

// Helper: Compute lighting at a point using Phong reflection model (Metal version)
float4 lighting_metal(Material material, PointLight light, float4 point, float4 eye_vector, float4 normal)
{
    // Combine material color with light intensity
    float4 effective_color = material.color * light.intensity;
    effective_color.w = 1.0;
    
    // Find direction to light
    float4 light_vector = normalize(light.position - point);
    
    // Compute ambient contribution
    float4 ambient = effective_color * material.ambient;
    ambient.w = 1.0;
    
    // light_dot_normal represents cosine of angle between light vector and normal
    float light_dot_normal = dot(light_vector, normal);
    
    float4 diffuse = float4(0, 0, 0, 0);
    float4 specular = float4(0, 0, 0, 0);
    
    if (light_dot_normal >= 0.0) {
        // Compute diffuse contribution
        diffuse = effective_color * material.diffuse * light_dot_normal;
        diffuse.w = 1.0;
        
        // Compute reflection vector
        float4 reflect_vector = -light_vector + normal * 2.0 * light_dot_normal;
        reflect_vector.w = 0.0;
        reflect_vector = normalize(reflect_vector);
        
        // reflect_dot_eye represents cosine of angle between reflection vector and eye
        float reflect_dot_eye = dot(reflect_vector, eye_vector);
        
        if (reflect_dot_eye > 0.0) {
            // Compute specular contribution
            float factor = pow(reflect_dot_eye, material.shininess);
            specular = light.intensity * material.specular * factor;
            specular.w = 1.0;
        }
    }
    
    // Add all three contributions
    float4 result = ambient + diffuse + specular;
    result.w = 1.0;
    return result;
}

// Helper: Find closest intersection t value
bool find_hit_t(float t0, float t1, thread float* hit_t)
{
    // Find the smallest positive t
    if (t0 > 0.0 && t1 > 0.0) {
        *hit_t = min(t0, t1);
        return true;
    } else if (t0 > 0.0) {
        *hit_t = t0;
        return true;
    } else if (t1 > 0.0) {
        *hit_t = t1;
        return true;
    }
    return false;
}

// Helper: Check if ray hits sphere and return hit distance
bool ray_sphere_intersect_detailed(Ray ray, Sphere sphere, thread float* hit_t)
{
    // Transform ray to object space
    float4x4 inverse_transform = float4x4(sphere.inverseTransform.columns[0],
                                          sphere.inverseTransform.columns[1],
                                          sphere.inverseTransform.columns[2],
                                          sphere.inverseTransform.columns[3]);
    Ray object_ray = transform_ray(ray, inverse_transform);
    
    // Intersection test
    float3 ray_origin = object_ray.origin.xyz;
    float3 ray_direction = object_ray.direction.xyz;
    
    float a = dot(ray_direction, ray_direction);
    float b = 2.0 * dot(ray_origin, ray_direction);
    float c = dot(ray_origin, ray_origin) - 1.0;
    
    float discriminant = b*b - 4.0*a*c;
    
    if (discriminant < 0.0) {
        return false;
    }
    
    float sqrt_disc = sqrt(discriminant);
    float t0 = (-b - sqrt_disc) / (2.0 * a);
    float t1 = (-b + sqrt_disc) / (2.0 * a);
    
    return find_hit_t(t0, t1, hit_t);
}

// Kernel: Render shaded sphere (Chapter 6)
// Full 3D rendering with Phong lighting model
kernel void render_sphere_shaded(texture2d<float, access::write> output [[ texture(0) ]],
                                 uint2 gid [[ thread_position_in_grid ]])
{
    // Canvas dimensions - increased for better quality (400x400 for sharper image)
    const int canvas_pixels = 400;
    
    // Check bounds
    if (gid.x >= canvas_pixels || gid.y >= canvas_pixels) {
        return;
    }
    
    // Ray origin (eye position)
    float4 ray_origin = float4(0.0, 0.0, -5.0, 1.0);
    
    // Wall position and size
    float wall_z = 10.0;
    float wall_size = 7.0;
    float half_size = wall_size / 2.0;
    float pixel_size = wall_size / canvas_pixels;
    
    // Compute world coordinates for this pixel
    // y increases downward in screen space, so we flip it
    float world_x = -half_size + pixel_size * gid.x;
    float world_y = half_size - pixel_size * gid.y;
    
    // Compute the point on the wall that the ray will target
    float4 wall_position = float4(world_x, world_y, wall_z, 1.0);
    
    // Create ray from origin toward wall - NORMALIZED DIRECTION
    float4 direction = normalize(wall_position - ray_origin);
    Ray ray;
    ray.origin = ray_origin;
    ray.direction = direction;
    
    // Create sphere with material
    Sphere sphere;
    sphere.id = 1;
    // Identity transform
    sphere.transform.columns[0] = float4(1.0, 0.0, 0.0, 0.0);
    sphere.transform.columns[1] = float4(0.0, 1.0, 0.0, 0.0);
    sphere.transform.columns[2] = float4(0.0, 0.0, 1.0, 0.0);
    sphere.transform.columns[3] = float4(0.0, 0.0, 0.0, 1.0);
    sphere.inverseTransform = sphere.transform;
    
    // Create light source
    PointLight light;
    light.position = float4(-10.0, 10.0, -10.0, 1.0);
    light.intensity = float4(1.0, 1.0, 1.0, 1.0);
    
    // Create material for sphere (magenta)
    Material material;
    material.color = float4(1.0, 0.2, 1.0, 1.0);
    material.ambient = 0.1;
    material.diffuse = 0.9;
    material.specular = 0.9;
    material.shininess = 200.0;
    
    // Check for intersection
    float hit_t;
    float4 color;
    
    if (ray_sphere_intersect_detailed(ray, sphere, &hit_t)) {
        // Hit! Compute lighting at intersection point
        
        // Compute hit point
        float4 point = ray.origin + ray.direction * hit_t;
        
        // Compute normal at hit point
        float4 normal = sphere_normal_at_metal(sphere, point);
        
        // Eye vector is the negative of ray direction
        float4 eye = -ray.direction;
        
        // Compute color with lighting
        color = lighting_metal(material, light, point, eye, normal);
    } else {
        // Miss - black background
        color = float4(0.0, 0.0, 0.0, 1.0);
    }
    
    // Write to output texture
    output.write(color, gid);
}

// Chapter 7: Render the full scene with multiple spheres
// This implements the scene from the end of Chapter 7 with 6 spheres
kernel void render_world_scene(texture2d<float, access::write> output [[ texture(0) ]],
                               uint2 gid [[ thread_position_in_grid ]])
{
    // Canvas size
    const int hsize = 400;
    const int vsize = 400;
    
    // Check bounds
    if (gid.x >= hsize || gid.y >= vsize) {
        return;
    }
    
    // Camera setup
    float fov = M_PI_F / 3.0;  // 60 degrees
    float half_view = tan(fov / 2.0);
    float aspect = float(hsize) / float(vsize);
    float half_width, half_height;
    
    if (aspect >= 1.0) {
        half_width = half_view;
        half_height = half_view / aspect;
    } else {
        half_width = half_view * aspect;
        half_height = half_view;
    }
    
    float pixel_size = (half_width * 2.0) / float(hsize);
    
    // Camera transform (view matrix)
    // Camera at (0, 1.5, -5) looking at (0, 1, 0)
    float4 from = float4(0.0, 1.5, -5.0, 1.0);
    float4 to = float4(0.0, 1.0, 0.0, 1.0);
    float4 upv = float4(0.0, 1.0, 0.0, 0.0);
    
    // View transform computation
    float4 forward = normalize(to - from);
    float4 upn = normalize(upv);
    float3 left3 = cross(forward.xyz, upn.xyz);
    float4 left = float4(left3.x, left3.y, left3.z, 0.0);
    float4 true_up = float4(cross(left.xyz, forward.xyz), 0.0);
    
    float4x4 orientation(left.x, true_up.x, -forward.x, 0.0,
                         left.y, true_up.y, -forward.y, 0.0,
                         left.z, true_up.z, -forward.z, 0.0,
                         0.0, 0.0, 0.0, 1.0);
    
    float4x4 translation(1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        -from.x, -from.y, -from.z, 1.0);
    
    float4x4 camera_transform = orientation * translation;
    float4x4 inv_camera = matrix_inverse_4x4(camera_transform);
    
    // Compute ray for this pixel
    float xoffset = (float(gid.x) + 0.5) * pixel_size;
    float yoffset = (float(gid.y) + 0.5) * pixel_size;
    
    float world_x = half_width - xoffset;
    float world_y = half_height - yoffset;
    
    float4 pixel = float4(world_x, world_y, -1.0, 1.0);
    float4 origin = float4(0.0, 0.0, 0.0, 1.0);
    
    float4 pixel_world = inv_camera * pixel;
    float4 origin_world = inv_camera * origin;
    
    float4 direction = normalize(pixel_world - origin_world);
    
    Ray ray;
    ray.origin = origin_world;
    ray.direction = direction;
    
    // Light source
    PointLight light;
    light.position = float4(-10.0, 10.0, -10.0, 1.0);
    light.intensity = float4(1.0, 1.0, 1.0, 1.0);
    
    // Materials
    Material floor_mat;
    floor_mat.color = float4(1.0, 0.9, 0.9, 1.0);
    floor_mat.ambient = 0.1;
    floor_mat.diffuse = 0.9;
    floor_mat.specular = 0.0;
    floor_mat.shininess = 200.0;
    
    Material middle_mat;
    middle_mat.color = float4(0.1, 1.0, 0.5, 1.0);
    middle_mat.ambient = 0.1;
    middle_mat.diffuse = 0.7;
    middle_mat.specular = 0.3;
    middle_mat.shininess = 200.0;
    
    Material right_mat;
    right_mat.color = float4(0.5, 1.0, 0.1, 1.0);
    right_mat.ambient = 0.1;
    right_mat.diffuse = 0.7;
    right_mat.specular = 0.3;
    right_mat.shininess = 200.0;
    
    Material left_mat;
    left_mat.color = float4(1.0, 0.8, 0.1, 1.0);
    left_mat.ambient = 0.1;
    left_mat.diffuse = 0.7;
    left_mat.specular = 0.3;
    left_mat.shininess = 200.0;
    
    // Scene objects - 6 spheres
    Sphere spheres[6];
    
    // 1. Floor - flattened sphere
    spheres[0].id = 1;
    float4x4 floor_transform = float4x4(10.0, 0.0, 0.0, 0.0,
                                         0.0, 0.01, 0.0, 0.0,
                                         0.0, 0.0, 10.0, 0.0,
                                         0.0, 0.0, 0.0, 1.0);
    spheres[0].transform.columns[0] = floor_transform[0];
    spheres[0].transform.columns[1] = floor_transform[1];
    spheres[0].transform.columns[2] = floor_transform[2];
    spheres[0].transform.columns[3] = floor_transform[3];
    float4x4 floor_inv = matrix_inverse_4x4(floor_transform);
    spheres[0].inverseTransform.columns[0] = floor_inv[0];
    spheres[0].inverseTransform.columns[1] = floor_inv[1];
    spheres[0].inverseTransform.columns[2] = floor_inv[2];
    spheres[0].inverseTransform.columns[3] = floor_inv[3];
    
    // 2. Left wall
    spheres[1].id = 2;
    float4x4 left_wall_transform = float4x4(1.0, 0.0, 0.0, 0.0,
                                              0.0, 1.0, 0.0, 0.0,
                                              0.0, 0.0, 1.0, 0.0,
                                              0.0, 0.0, 5.0, 1.0) *
                                   float4x4(cos(M_PI_F/4.0), 0.0, sin(M_PI_F/4.0), 0.0,
                                           0.0, 1.0, 0.0, 0.0,
                                           -sin(M_PI_F/4.0), 0.0, cos(M_PI_F/4.0), 0.0,
                                           0.0, 0.0, 0.0, 1.0) *
                                   float4x4(1.0, 0.0, 0.0, 0.0,
                                           0.0, cos(M_PI_F/2.0), -sin(M_PI_F/2.0), 0.0,
                                           0.0, sin(M_PI_F/2.0), cos(M_PI_F/2.0), 0.0,
                                           0.0, 0.0, 0.0, 1.0) *
                                   float4x4(10.0, 0.0, 0.0, 0.0,
                                           0.0, 0.01, 0.0, 0.0,
                                           0.0, 0.0, 10.0, 0.0,
                                           0.0, 0.0, 0.0, 1.0);
    spheres[1].transform.columns[0] = left_wall_transform[0];
    spheres[1].transform.columns[1] = left_wall_transform[1];
    spheres[1].transform.columns[2] = left_wall_transform[2];
    spheres[1].transform.columns[3] = left_wall_transform[3];
    float4x4 left_wall_inv = matrix_inverse_4x4(left_wall_transform);
    spheres[1].inverseTransform.columns[0] = left_wall_inv[0];
    spheres[1].inverseTransform.columns[1] = left_wall_inv[1];
    spheres[1].inverseTransform.columns[2] = left_wall_inv[2];
    spheres[1].inverseTransform.columns[3] = left_wall_inv[3];
    
    // 3. Right wall
    spheres[2].id = 3;
    float4x4 right_wall_transform = float4x4(1.0, 0.0, 0.0, 0.0,
                                               0.0, 1.0, 0.0, 0.0,
                                               0.0, 0.0, 1.0, 0.0,
                                               0.0, 0.0, 5.0, 1.0) *
                                    float4x4(cos(-M_PI_F/4.0), 0.0, sin(-M_PI_F/4.0), 0.0,
                                            0.0, 1.0, 0.0, 0.0,
                                            -sin(-M_PI_F/4.0), 0.0, cos(-M_PI_F/4.0), 0.0,
                                            0.0, 0.0, 0.0, 1.0) *
                                    float4x4(1.0, 0.0, 0.0, 0.0,
                                            0.0, cos(M_PI_F/2.0), -sin(M_PI_F/2.0), 0.0,
                                            0.0, sin(M_PI_F/2.0), cos(M_PI_F/2.0), 0.0,
                                            0.0, 0.0, 0.0, 1.0) *
                                    float4x4(10.0, 0.0, 0.0, 0.0,
                                            0.0, 0.01, 0.0, 0.0,
                                            0.0, 0.0, 10.0, 0.0,
                                            0.0, 0.0, 0.0, 1.0);
    spheres[2].transform.columns[0] = right_wall_transform[0];
    spheres[2].transform.columns[1] = right_wall_transform[1];
    spheres[2].transform.columns[2] = right_wall_transform[2];
    spheres[2].transform.columns[3] = right_wall_transform[3];
    float4x4 right_wall_inv = matrix_inverse_4x4(right_wall_transform);
    spheres[2].inverseTransform.columns[0] = right_wall_inv[0];
    spheres[2].inverseTransform.columns[1] = right_wall_inv[1];
    spheres[2].inverseTransform.columns[2] = right_wall_inv[2];
    spheres[2].inverseTransform.columns[3] = right_wall_inv[3];
    
    // 4. Middle sphere - green, slightly up
    spheres[3].id = 4;
    float4x4 middle_transform = float4x4(1.0, 0.0, 0.0, 0.0,
                                           0.0, 1.0, 0.0, 0.0,
                                           0.0, 0.0, 1.0, 0.0,
                                           -0.5, 1.0, 0.5, 1.0);
    spheres[3].transform.columns[0] = middle_transform[0];
    spheres[3].transform.columns[1] = middle_transform[1];
    spheres[3].transform.columns[2] = middle_transform[2];
    spheres[3].transform.columns[3] = middle_transform[3];
    float4x4 middle_inv = matrix_inverse_4x4(middle_transform);
    spheres[3].inverseTransform.columns[0] = middle_inv[0];
    spheres[3].inverseTransform.columns[1] = middle_inv[1];
    spheres[3].inverseTransform.columns[2] = middle_inv[2];
    spheres[3].inverseTransform.columns[3] = middle_inv[3];
    
    // 5. Right sphere - smaller, green, scaled
    spheres[4].id = 5;
    float4x4 right_transform = float4x4(1.0, 0.0, 0.0, 0.0,
                                        0.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 1.0, 0.0,
                                        1.5, 0.5, -0.5, 1.0) *
                             float4x4(0.5, 0.0, 0.0, 0.0,
                                     0.0, 0.5, 0.0, 0.0,
                                     0.0, 0.0, 0.5, 0.0,
                                     0.0, 0.0, 0.0, 1.0);
    spheres[4].transform.columns[0] = right_transform[0];
    spheres[4].transform.columns[1] = right_transform[1];
    spheres[4].transform.columns[2] = right_transform[2];
    spheres[4].transform.columns[3] = right_transform[3];
    float4x4 right_inv = matrix_inverse_4x4(right_transform);
    spheres[4].inverseTransform.columns[0] = right_inv[0];
    spheres[4].inverseTransform.columns[1] = right_inv[1];
    spheres[4].inverseTransform.columns[2] = right_inv[2];
    spheres[4].inverseTransform.columns[3] = right_inv[3];
    
    // 6. Left sphere - smallest, yellow
    spheres[5].id = 6;
    float4x4 left_transform = float4x4(1.0, 0.0, 0.0, 0.0,
                                         0.0, 1.0, 0.0, 0.0,
                                         0.0, 0.0, 1.0, 0.0,
                                         -1.5, 0.33, -0.75, 1.0) *
                            float4x4(0.33, 0.0, 0.0, 0.0,
                                    0.0, 0.33, 0.0, 0.0,
                                    0.0, 0.0, 0.33, 0.0,
                                    0.0, 0.0, 0.0, 1.0);
    spheres[5].transform.columns[0] = left_transform[0];
    spheres[5].transform.columns[1] = left_transform[1];
    spheres[5].transform.columns[2] = left_transform[2];
    spheres[5].transform.columns[3] = left_transform[3];
    float4x4 left_inv = matrix_inverse_4x4(left_transform);
    spheres[5].inverseTransform.columns[0] = left_inv[0];
    spheres[5].inverseTransform.columns[1] = left_inv[1];
    spheres[5].inverseTransform.columns[2] = left_inv[2];
    spheres[5].inverseTransform.columns[3] = left_inv[3];
    
    // Find closest intersection
    float closest_t = INFINITY;
    int closest_sphere = -1;
    
    for (int i = 0; i < 6; i++) {
        float t0, t1;
        if (ray_sphere_intersect_detailed(ray, spheres[i], &t0)) {
            if (t0 > 0.0 && t0 < closest_t) {
                closest_t = t0;
                closest_sphere = i;
            }
        }
    }
    
    float4 color;
    
    if (closest_sphere >= 0) {
        // Hit! Compute lighting
        Sphere hit_sphere = spheres[closest_sphere];
        float4 point = ray.origin + ray.direction * closest_t;
        float4 normal = sphere_normal_at_metal(hit_sphere, point);
        float4 eye = -ray.direction;
        
        // Select material based on sphere
        Material mat;
        if (closest_sphere <= 2) {
            mat = floor_mat;  // Floor and walls
        } else if (closest_sphere == 3) {
            mat = middle_mat;
        } else if (closest_sphere == 4) {
            mat = right_mat;
        } else {
            mat = left_mat;
        }
        
        color = lighting_metal(mat, light, point, eye, normal);
    } else {
        // Miss - black background
        color = float4(0.0, 0.0, 0.0, 1.0);
    }
    
    output.write(color, gid);
}
