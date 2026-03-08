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
    constexpr sampler textureSampler (mag_filter::nearest, min_filter::nearest);
    
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

// Chapter 8: Shadows

// Forward declaration
bool ray_sphere_intersect_detailed(Ray ray, Sphere sphere, thread float* hit_t);

// Helper: Check if a point is in shadow (Metal version)
// Casts a ray from point toward light, returns true if anything blocks it
bool is_shadowed_metal(Sphere spheres[6], int sphere_count, PointLight light, float4 point)
{
    // Vector from point to light
    float4 light_dir = light.position - point;
    float distance = length(light_dir.xyz);
    light_dir = normalize(light_dir);
    
    // Create shadow ray starting slightly offset from point
    Ray shadow_ray;
    shadow_ray.origin = point + light_dir * 0.001;  // Small offset to avoid self-intersection
    shadow_ray.direction = light_dir;
    
    // Check for intersections with all spheres
    for (int i = 0; i < sphere_count; i++) {
        float t0, t1;
        if (ray_sphere_intersect_detailed(shadow_ray, spheres[i], &t0)) {
            // Check if intersection is between point and light
            if (t0 > 0.0 && t0 < distance) {
                return true;  // In shadow
            }
        }
    }
    
    return false;  // Not in shadow
}

// Helper: Compute lighting with shadow support (Metal version)
float4 lighting_metal_shadow(Material material, PointLight light, float4 point, float4 eye_vector, float4 normal, bool in_shadow)
{
    // Combine material color with light intensity
    float4 effective_color = material.color * light.intensity;
    effective_color.w = 1.0;
    
    // Find direction to light
    float4 light_vector = normalize(light.position - point);
    
    // Compute ambient contribution (always present, even in shadow)
    float4 ambient = effective_color * material.ambient;
    ambient.w = 1.0;
    
    // If in shadow, only return ambient light
    if (in_shadow) {
        float4 result = ambient;
        result.w = 1.0;
        return result;
    }
    
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

// Chapter 9: Plane intersection functions for Metal

// Helper: Ray-plane intersection
// Returns true if hit, false if miss. Writes t value if hit.
bool intersect_plane_metal(Ray ray, Plane plane, thread float* hit_t)
{
    // Transform ray to plane's object space
    float4x4 inv = float4x4(plane.inverseTransform.columns[0],
                           plane.inverseTransform.columns[1],
                           plane.inverseTransform.columns[2],
                           plane.inverseTransform.columns[3]);
    Ray object_ray = transform_ray(ray, inv);
    
    // Check if ray is parallel to plane (direction.y is nearly 0)
    if (abs(object_ray.direction.y) < 0.0001) {
        return false;
    }
    
    // Compute intersection: solve for t when y = 0
    // origin.y + direction.y * t = 0
    // t = -origin.y / direction.y
    float t = -object_ray.origin.y / object_ray.direction.y;
    
    if (t > 0.0) {
        *hit_t = t;
        return true;
    }
    
    return false;
}

// Helper: Compute normal at point on plane
float4 plane_normal_at_metal(Plane plane, float4 point)
{
    // Transform to object space
    float4x4 inv = float4x4(plane.inverseTransform.columns[0],
                           plane.inverseTransform.columns[1],
                           plane.inverseTransform.columns[2],
                           plane.inverseTransform.columns[3]);
    
    // Normal in object space is (0, 1, 0)
    float4 object_normal = float4(0.0, 1.0, 0.0, 0.0);
    
    // Transform to world space using inverse transpose
    float4x4 inv_transpose = transpose(inv);
    float4 world_normal = inv_transpose * object_normal;
    world_normal.w = 0.0;
    
    return normalize(world_normal);
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

// Chapter 9: Hexagonal Room Demo
// A room with 6 walls (planes) arranged in a hexagon, viewed from above
kernel void render_hexagonal_room(texture2d<float, access::write> output [[ texture(0) ]],
                                  uint2 gid [[ thread_position_in_grid ]])
{
    const int hsize = 1920;
    const int vsize = 1080;
    
    if (gid.x >= hsize || gid.y >= vsize) return;
    
    // Bird's eye view - map pixels to world coordinates
    float view_size = 20.0;
    float aspect = float(hsize) / float(vsize);
    
    float u = (float(gid.x) / float(hsize)) * 2.0 - 1.0;
    float v = (float(gid.y) / float(vsize)) * 2.0 - 1.0;
    
    float world_x = u * view_size * aspect;
    float world_z = v * view_size;
    
    // Hexagon parameters
    float room_radius = 8.0;
    float dist = sqrt(world_x * world_x + world_z * world_z);
    float angle = atan2(world_z, world_x);
    if (angle < 0.0) angle += 2.0 * M_PI_F;
    
    float sector = M_PI_F / 3.0;
    float angle_in_sector = fmod(angle, sector);
    if (angle_in_sector > sector / 2.0) angle_in_sector = sector - angle_in_sector;
    float max_dist = room_radius / cos(angle_in_sector);
    
    float4 color;
    
    if (dist < max_dist) {
        // Inside - checkerboard floor
        float check_x = floor((world_x + 100.0) / 2.0);
        float check_z = floor((world_z + 100.0) / 2.0);
        float pattern = fmod(check_x + check_z, 2.0);
        
        if (pattern < 0.5) {
            color = float4(0.4, 0.3, 0.2, 1.0);  // Dark wood
        } else {
            color = float4(0.5, 0.4, 0.3, 1.0);  // Light wood
        }
        
        // Walls at perimeter
        if (dist > max_dist * 0.9) {
            color = float4(0.8, 0.7, 0.6, 1.0);
        }
    } else {
        // Outside
        color = float4(0.05, 0.05, 0.1, 1.0);
    }
    
    output.write(color, gid);
}

// Chapter 10: Pattern rendering demo
// Renders spheres with different patterns: stripes, gradient, ring, checker
kernel void render_patterns_demo(texture2d<float, access::write> output [[ texture(0) ]],
                                 uint2 gid [[ thread_position_in_grid ]])
{
    const int hsize = 1920;
    const int vsize = 1080;
    
    if (gid.x >= hsize || gid.y >= vsize) return;
    
    // Camera at z=-8 shooting toward wall at z=5 (closer wall)
    float4 ray_origin = float4(0, 1, -8, 1);
    float wall_z = 5.0;
    float wall_size = 10.0;
    float pixel_size = wall_size / vsize;
    
    float world_x = -wall_size/2.0 + pixel_size * gid.x;
    float world_y = wall_size/2.0 - pixel_size * gid.y;
    float4 wall_pos = float4(world_x, world_y, wall_z, 1);
    
    float4 ray_direction = normalize(wall_pos - ray_origin);
    
    Ray ray;
    ray.origin = ray_origin;
    ray.direction = ray_direction;
    
    // Light
    PointLight light;
    light.position = float4(-10, 10, -10, 1);
    light.intensity = float4(1, 1, 1, 1);
    
    // 4 spheres with patterns
    Sphere spheres[4];
    float4 white = float4(1, 1, 1, 1);
    float4 black = float4(0.1, 0.1, 0.1, 1);
    float4 red = float4(1, 0.3, 0.3, 1);
    float4 blue = float4(0.3, 0.3, 1, 1);
    
    // Helper to create a sphere with transform
    auto make_sphere = [&](int id, float tx, float ty, float tz, float scale) -> Sphere {
        Sphere s;
        s.id = id;
        float4x4 mat = float4x4(scale, 0, 0, 0,
                                0, scale, 0, 0,
                                0, 0, scale, 0,
                                tx, ty, tz, 1);
        s.transform.columns[0] = mat[0];
        s.transform.columns[1] = mat[1];
        s.transform.columns[2] = mat[2];
        s.transform.columns[3] = mat[3];
        float4x4 inv = matrix_inverse_4x4(mat);
        s.inverseTransform.columns[0] = inv[0];
        s.inverseTransform.columns[1] = inv[1];
        s.inverseTransform.columns[2] = inv[2];
        s.inverseTransform.columns[3] = inv[3];
        return s;
    };
    
    // Spheres arranged left to right, closer together
    spheres[0] = make_sphere(1, -2.5, 1, 0, 0.5);  // Stripe (leftmost)
    spheres[1] = make_sphere(2, -0.8, 1, 0, 0.5);   // Gradient
    spheres[2] = make_sphere(3, 0.8, 1, 0, 0.5);   // Ring
    spheres[3] = make_sphere(4, 2.5, 1, 0, 0.5);   // Checker (rightmost)
    
    // Find closest hit
    float closest_t = 999999.0;
    int hit_id = -1;
    float4 hit_point, hit_normal;
    
    for (int i = 0; i < 4; i++) {
        float t;
        if (ray_sphere_intersect_detailed(ray, spheres[i], &t)) {
            if (t > 0.001 && t < closest_t) {
                closest_t = t;
                hit_id = spheres[i].id;
                hit_point = ray.origin + ray.direction * t;
                hit_normal = sphere_normal_at_metal(spheres[i], hit_point);
            }
        }
    }
    
    float4 color;
    
    if (hit_id != -1) {
        float4 eye = -ray.direction;
        Sphere hit_sphere = spheres[hit_id - 1];
        
        // Transform to object space
        float4x4 inv = float4x4(hit_sphere.inverseTransform.columns[0],
                                hit_sphere.inverseTransform.columns[1],
                                hit_sphere.inverseTransform.columns[2],
                                hit_sphere.inverseTransform.columns[3]);
        float4 obj_pt = inv * hit_point;
        
        float4 pat_col;
        switch (hit_id) {
            case 1: { // Stripe
                float stripe_pt = obj_pt.x * 2.0;
                int idx = int(floor(abs(stripe_pt)));
                pat_col = (idx % 2 == 0) ? white : black;
                break;
            }
            case 2: { // Gradient
                float dist = obj_pt.x - floor(obj_pt.x);
                pat_col = red + (blue - red) * dist;
                pat_col.w = 1.0;
                break;
            }
            case 3: { // Ring - concentric circles around Z axis (visible from front)
                // Rings based on distance from Z axis in XY plane
                float dist = sqrt(obj_pt.x * obj_pt.x + obj_pt.y * obj_pt.y);
                int idx = int(floor(dist * 3.0)); // Scale to get ~3-4 rings
                pat_col = (idx % 2 == 0) ? white : black;
                break;
            }
            case 4: { // Checker
                int xi = int(floor(obj_pt.x));
                int yi = int(floor(obj_pt.y));
                int zi = int(floor(obj_pt.z));
                pat_col = ((xi + yi + zi) % 2 == 0) ? white : black;
                break;
            }
            default: pat_col = white;
        }
        
        Material mat;
        mat.color = pat_col;
        mat.ambient = 0.1;
        mat.diffuse = 0.9;
        mat.specular = 0.9;
        mat.shininess = 200.0;
        
        bool shadow = is_shadowed_metal(spheres, 4, light, hit_point);
        color = lighting_metal_shadow(mat, light, hit_point, eye, hit_normal, shadow);
    } else {
        // Sky gradient
        float t = float(gid.y) / float(vsize);
        color = float4(0.05, 0.05, 0.1 + 0.1 * t, 1.0);
    }
    
    output.write(color, gid);
}

// Chapter 11: Reflection & Refraction demo with animated camera
// Renders a scene with a mirror sphere, glass sphere, and colored spheres
// Camera pans back and forth to show real-time rendering speed
kernel void render_reflection_refraction_demo(texture2d<float, access::write> output [[ texture(0) ]],
                                               constant float &time [[ buffer(1) ]],
                                               uint2 gid [[ thread_position_in_grid ]])
{
    const int hsize = 1920;
    const int vsize = 1080;
    
    if (gid.x >= hsize || gid.y >= vsize) return;
    
    // Camera animation: pans back and forth in an arc
    // Time parameter drives the camera position
    float angle = sin(time * 0.5) * 0.3;  // Oscillates between -0.3 and 0.3 radians
    float cam_x = sin(angle) * 8.0;  // X position oscillates
    float cam_z = -cos(angle) * 8.0;   // Z position stays negative (looking at scene)
    
    float4 ray_origin = float4(cam_x, 1.5, cam_z, 1);
    float wall_z = 5.0;
    float wall_size = 10.0;
    float pixel_size = wall_size / vsize;
    
    float world_x = -wall_size/2.0 + pixel_size * gid.x;
    float world_y = wall_size/2.0 - pixel_size * gid.y;
    float4 wall_pos = float4(world_x, world_y, wall_z, 1);
    
    float4 ray_direction = normalize(wall_pos - ray_origin);
    
    Ray ray;
    ray.origin = ray_origin;
    ray.direction = ray_direction;
    
    // Light
    PointLight light;
    light.position = float4(-10, 15, -10, 1);
    light.intensity = float4(1, 1, 1, 1);
    
    // Colors
    float4 white = float4(1, 1, 1, 1);
    float4 red = float4(1, 0.2, 0.2, 1);
    float4 green = float4(0.2, 1, 0.2, 1);
    float4 blue = float4(0.2, 0.2, 1, 1);
    float4 yellow = float4(1, 1, 0.2, 1);
    
    // Create 6 spheres: 2 special (mirror, glass), 4 colored
    Sphere spheres[6];
    Material materials[6];
    
    // Helper to create a sphere with transform
    auto make_sphere = [&](int id, float tx, float ty, float tz, float scale) -> Sphere {
        Sphere s;
        s.id = id;
        float4x4 mat = float4x4(scale, 0, 0, 0,
                                0, scale, 0, 0,
                                0, 0, scale, 0,
                                tx, ty, tz, 1);
        s.transform.columns[0] = mat[0];
        s.transform.columns[1] = mat[1];
        s.transform.columns[2] = mat[2];
        s.transform.columns[3] = mat[3];
        float4x4 inv = matrix_inverse_4x4(mat);
        s.inverseTransform.columns[0] = inv[0];
        s.inverseTransform.columns[1] = inv[1];
        s.inverseTransform.columns[2] = inv[2];
        s.inverseTransform.columns[3] = inv[3];
        return s;
    };
    
    // Sphere 1: Mirror sphere (left, highly reflective)
    spheres[0] = make_sphere(1, -2.0, 1.0, 0, 0.8);
    materials[0].color = white;
    materials[0].ambient = 0.1;
    materials[0].diffuse = 0.1;
    materials[0].specular = 1.0;
    materials[0].shininess = 300.0;
    materials[0].reflective = 0.95;  // Almost perfect mirror
    materials[0].transparency = 0.0;
    materials[0].refractive_index = 1.0;
    
    // Sphere 2: Glass sphere (center, transparent)
    spheres[1] = make_sphere(2, 0, 1.0, 0, 0.8);
    materials[1].color = white;
    materials[1].ambient = 0.05;
    materials[1].diffuse = 0.05;
    materials[1].specular = 1.0;
    materials[1].shininess = 300.0;
    materials[1].reflective = 0.1;  // Slight reflection
    materials[1].transparency = 0.9;  // Mostly transparent
    materials[1].refractive_index = 1.5;  // Glass
    
    // Sphere 3: Red sphere (right)
    spheres[2] = make_sphere(3, 2.0, 1.0, 0, 0.8);
    materials[2].color = red;
    materials[2].ambient = 0.1;
    materials[2].diffuse = 0.9;
    materials[2].specular = 0.9;
    materials[2].shininess = 200.0;
    materials[2].reflective = 0.0;
    materials[2].transparency = 0.0;
    materials[2].refractive_index = 1.0;
    
    // Sphere 4: Green sphere (back left)
    spheres[3] = make_sphere(4, -1.5, 0.5, 2.0, 0.6);
    materials[3].color = green;
    materials[3].ambient = 0.1;
    materials[3].diffuse = 0.9;
    materials[3].specular = 0.9;
    materials[3].shininess = 200.0;
    materials[3].reflective = 0.0;
    materials[3].transparency = 0.0;
    materials[3].refractive_index = 1.0;
    
    // Sphere 5: Blue sphere (back right)
    spheres[4] = make_sphere(5, 1.5, 0.5, 2.0, 0.6);
    materials[4].color = blue;
    materials[4].ambient = 0.1;
    materials[4].diffuse = 0.9;
    materials[4].specular = 0.9;
    materials[4].shininess = 200.0;
    materials[4].reflective = 0.0;
    materials[4].transparency = 0.0;
    materials[4].refractive_index = 1.0;
    
    // Sphere 6: Yellow sphere (front)
    spheres[5] = make_sphere(6, 0, 0.5, -1.5, 0.5);
    materials[5].color = yellow;
    materials[5].ambient = 0.1;
    materials[5].diffuse = 0.9;
    materials[5].specular = 0.9;
    materials[5].shininess = 200.0;
    materials[5].reflective = 0.0;
    materials[5].transparency = 0.0;
    materials[5].refractive_index = 1.0;
    
    // Find closest hit
    float closest_t = 999999.0;
    int hit_id = -1;
    float4 hit_point, hit_normal;
    
    for (int i = 0; i < 6; i++) {
        float t;
        if (ray_sphere_intersect_detailed(ray, spheres[i], &t)) {
            if (t > 0.001 && t < closest_t) {
                closest_t = t;
                hit_id = spheres[i].id;
                hit_point = ray.origin + ray.direction * t;
                hit_normal = sphere_normal_at_metal(spheres[i], hit_point);
            }
        }
    }
    
    float4 color;
    
    if (hit_id != -1) {
        int idx = hit_id - 1;
        Sphere hit_sphere = spheres[idx];
        Material hit_material = materials[idx];
        
        float4 eye = -ray.direction;
        
        // Base lighting
        bool shadow = is_shadowed_metal(spheres, 6, light, hit_point);
        float4 base_color = lighting_metal_shadow(hit_material, light, hit_point, eye, hit_normal, shadow);
        
        // Reflection (simplified - just one level)
        float4 reflected = float4(0, 0, 0, 1);
        if (hit_material.reflective > 0.0) {
            float4 reflect_dir = reflect(ray.direction, hit_normal);
            reflect_dir.w = 0.0;
            reflect_dir = normalize(reflect_dir);
            
            Ray reflect_ray;
            reflect_ray.origin = hit_point + hit_normal * 0.001;
            reflect_ray.direction = reflect_dir;
            
            // Cast reflection ray and find what it hits
            float reflect_closest = 999999.0;
            int reflect_hit_id = -1;
            float4 reflect_hit_point, reflect_hit_normal;
            
            for (int i = 0; i < 6; i++) {
                float t;
                if (ray_sphere_intersect_detailed(reflect_ray, spheres[i], &t)) {
                    if (t > 0.001 && t < reflect_closest && i != idx) {  // Don't hit self
                        reflect_closest = t;
                        reflect_hit_id = spheres[i].id;
                        reflect_hit_point = reflect_ray.origin + reflect_ray.direction * t;
                        reflect_hit_normal = sphere_normal_at_metal(spheres[i], reflect_hit_point);
                    }
                }
            }
            
            if (reflect_hit_id != -1) {
                int ridx = reflect_hit_id - 1;
                Material rmat = materials[ridx];
                bool rshadow = is_shadowed_metal(spheres, 6, light, reflect_hit_point);
                float4 reye = -reflect_ray.direction;
                reflected = lighting_metal_shadow(rmat, light, reflect_hit_point, reye, reflect_hit_normal, rshadow);
            } else {
                // Sky reflection
                reflected = float4(0.05, 0.05, 0.15, 1.0);
            }
        }
        
        // Combine: base + reflection
        float refl = hit_material.reflective;
        color = base_color * (1.0 - refl) + reflected * refl;
        color.w = 1.0;
    } else {
        // Sky gradient
        float t = float(gid.y) / float(vsize);
        color = float4(0.05, 0.05, 0.15 + 0.1 * t, 1.0);
    }
    
    output.write(color, gid);
}
