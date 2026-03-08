#import <XCTest/XCTest.h>
#import "SharedTypes.h"

@interface ShadowTests : XCTestCase
@end

@implementation ShadowTests

// Helper to check if two colors are approximately equal
- (BOOL)color:(vector_float4)a equalsColor:(vector_float4)b tolerance:(float)tol {
    return simd_all(simd_abs(a - b) < tol);
}

// Chapter 8 - Lighting with surface in shadow
- (void)testLightingWithSurfaceInShadow {
    Material m = DEFAULT_MATERIAL;
    vector_float4 position = {0, 0, 0, 1};
    
    vector_float4 eye_vector = {0, 0, -1, 0};
    vector_float4 normal = {0, 0, -1, 0};
    PointLight light = point_light_create((vector_float4){0, 0, -10, 1}, 
                                          (vector_float4){1, 1, 1, 1});
    
    int in_shadow = 1;  // In shadow
    
    vector_float4 result = lighting_with_shadow(m, light, position, eye_vector, normal, in_shadow);
    vector_float4 expected = {0.1, 0.1, 0.1, 1};  // Only ambient
    
    XCTAssertTrue([self color:result equalsColor:expected tolerance:0.0001]);
}

// Chapter 8 - There is no shadow when nothing is collinear with point and light
- (void)testNoShadowWhenNothingBlocksLight {
    World w = world_create_default();
    vector_float4 point = {0, 10, 0, 1};
    
    int shadowed = is_shadowed(w, point);
    
    XCTAssertEqual(shadowed, 0);
}

// Chapter 8 - The shadow when an object is between the point and the light
- (void)testShadowWhenObjectBlocksLight {
    World w = world_create_default();
    vector_float4 point = {10, -10, 10, 1};
    
    int shadowed = is_shadowed(w, point);
    
    XCTAssertEqual(shadowed, 1);
}

// Chapter 8 - There is no shadow when an object is behind the light
- (void)testNoShadowWhenObjectBehindLight {
    World w = world_create_default();
    vector_float4 point = {-20, 20, -20, 1};
    
    int shadowed = is_shadowed(w, point);
    
    XCTAssertEqual(shadowed, 0);
}

// Chapter 8 - There is no shadow when an object is behind the point
- (void)testNoShadowWhenObjectBehindPoint {
    World w = world_create_default();
    vector_float4 point = {-2, 2, -2, 1};
    
    int shadowed = is_shadowed(w, point);
    
    XCTAssertEqual(shadowed, 0);
}

// Chapter 8 - shade_hit() is given an intersection in shadow
- (void)testShadeHitGivenIntersectionInShadow {
    World w = world_create_empty();
    w.light = point_light_create((vector_float4){0, 0, -10, 1}, 
                                  (vector_float4){1, 1, 1, 1});
    
    // Add two spheres
    Sphere s1 = sphere_create(1);
    world_add_sphere(&w, s1);
    
    Sphere s2 = sphere_create(2);
    s2.transform = matrix_translation(0, 0, 10);
    matrix_float4x4 mat = matrix_from_columns(s2.transform.columns[0],
                                              s2.transform.columns[1],
                                              s2.transform.columns[2],
                                              s2.transform.columns[3]);
    matrix_float4x4 inv = matrix_invert(mat);
    s2.inverseTransform.columns[0] = inv.columns[0];
    s2.inverseTransform.columns[1] = inv.columns[1];
    s2.inverseTransform.columns[2] = inv.columns[2];
    s2.inverseTransform.columns[3] = inv.columns[3];
    world_add_sphere(&w, s2);
    
    // Ray that hits the first sphere
    Ray r = ray_create((vector_float4){0, 0, 5, 1}, (vector_float4){0, 0, 1, 0});
    
    Intersection xs[MAX_INTERSECTIONS_TOTAL];
    int count = intersect_world(w, r, xs);
    Intersection hit = hit_from_array(xs, count);
    
    vector_float4 c = shade_hit(w, hit, r);
    
    // Should be dim (ambient only)
    XCTAssertLessThan(c.x, 0.2);
    XCTAssertLessThan(c.y, 0.2);
    XCTAssertLessThan(c.z, 0.2);
}

// Chapter 8 - The hit should offset the point
- (void)testHitShouldOffsetPoint {
    Ray r = ray_create((vector_float4){0, 0, -5, 1}, (vector_float4){0, 0, 1, 0});
    Sphere s = sphere_create(1);
    s.transform = matrix_translation(0, 0, 1);
    matrix_float4x4 mat = matrix_from_columns(s.transform.columns[0],
                                              s.transform.columns[1],
                                              s.transform.columns[2],
                                              s.transform.columns[3]);
    matrix_float4x4 inv = matrix_invert(mat);
    s.inverseTransform.columns[0] = inv.columns[0];
    s.inverseTransform.columns[1] = inv.columns[1];
    s.inverseTransform.columns[2] = inv.columns[2];
    s.inverseTransform.columns[3] = inv.columns[3];
    
    Intersection i;
    i.t = 5;
    i.objectId = 1;
    
    // The point should be offset from the surface to avoid self-intersection
    vector_float4 point = ray_position(r, i.t);
    vector_float4 normal = sphere_normal_at(s, point);
    vector_float4 offset_point = point + normal * 0.001f;
    
    // Offset point should be slightly above surface
    float distance = simd_length((offset_point - point).xyz);
    XCTAssertEqualWithAccuracy(distance, 0.001f, 0.0001f);
}

@end
