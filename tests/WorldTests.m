#import <XCTest/XCTest.h>
#import "SharedTypes.h"

@interface WorldTests : XCTestCase
@end

@implementation WorldTests

// Helper to check if two tuples are approximately equal
- (BOOL)tuple:(vector_float4)a equalsTuple:(vector_float4)b tolerance:(float)tol {
    return simd_all(simd_abs(a - b) < tol);
}

// Helper to check if two colors are approximately equal
- (BOOL)color:(vector_float4)a equalsColor:(vector_float4)b tolerance:(float)tol {
    return simd_all(simd_abs(a - b) < tol);
}

// Chapter 7 - Creating a world
- (void)testCreatingWorld {
    World w = world_create_empty();
    
    XCTAssertEqual(w.sphere_count, 0);
}

// Chapter 7 - The default world
- (void)testDefaultWorld {
    World w = world_create_default();
    
    // Should have a light
    vector_float4 expected_light_pos = {-10, 10, -10, 1};
    vector_float4 expected_light_intensity = {1, 1, 1, 1};
    XCTAssertTrue([self tuple:w.light.position equalsTuple:expected_light_pos tolerance:0.0001]);
    XCTAssertTrue([self tuple:w.light.intensity equalsTuple:expected_light_intensity tolerance:0.0001]);
    
    // Should have two spheres
    XCTAssertEqual(w.sphere_count, 2);
}

// Chapter 7 - Intersect a world with a ray
- (void)testIntersectWorldWithRay {
    World w = world_create_default();
    
    Ray r = ray_create((vector_float4){0, 0, -5, 1}, (vector_float4){0, 0, 1, 0});
    
    Intersection xs[MAX_INTERSECTIONS_TOTAL];
    int count = intersect_world(w, r, xs);
    
    // Should have 4 intersections (2 spheres * 2 intersections each)
    XCTAssertEqual(count, 4);
    
    // First sphere is at scale 0.5, so intersections at t=4 and t=4.5
    // Second sphere is at origin, so intersections at t=5.5 and t=6
    XCTAssertEqualWithAccuracy(xs[0].t, 4.0, 0.0001);
    XCTAssertEqualWithAccuracy(xs[1].t, 4.5, 0.0001);
    XCTAssertEqualWithAccuracy(xs[2].t, 5.5, 0.0001);
    XCTAssertEqualWithAccuracy(xs[3].t, 6.0, 0.0001);
}

// Chapter 7 - Shading an intersection
- (void)testShadingIntersection {
    World w = world_create_default();
    
    Ray r = ray_create((vector_float4){0, 0, -5, 1}, (vector_float4){0, 0, 1, 0});
    
    Intersection xs[MAX_INTERSECTIONS_TOTAL];
    int count = intersect_world(w, r, xs);
    
    Intersection hit = hit_from_array(xs, count);
    
    vector_float4 c = shade_hit(w, hit, r);
    
    // Should get a color (not black)
    // The exact color depends on lighting calculations
    XCTAssertGreaterThan(c.x + c.y + c.z, 0.0);
}

// Chapter 7 - Shading an intersection from the inside
- (void)testShadingIntersectionFromInside {
    World w = world_create_default();
    
    // Ray starting inside the inner sphere
    Ray r = ray_create((vector_float4){0, 0, 0, 1}, (vector_float4){0, 0, 1, 0});
    
    Intersection xs[MAX_INTERSECTIONS_TOTAL];
    int count = intersect_world(w, r, xs);
    
    Intersection hit = hit_from_array(xs, count);
    
    vector_float4 c = shade_hit(w, hit, r);
    
    // Should still get a color
    XCTAssertGreaterThan(c.x + c.y + c.z, 0.0);
}

// Chapter 7 - Color when a ray misses
- (void)testColorWhenRayMisses {
    World w = world_create_default();
    
    Ray r = ray_create((vector_float4){0, 0, -5, 1}, (vector_float4){0, 1, 0, 0});
    
    vector_float4 c = color_at(w, r);
    vector_float4 black = {0, 0, 0, 1};
    
    XCTAssertTrue([self color:c equalsColor:black tolerance:0.0001]);
}

// Chapter 7 - Color when a ray hits
- (void)testColorWhenRayHits {
    World w = world_create_default();
    
    Ray r = ray_create((vector_float4){0, 0, -5, 1}, (vector_float4){0, 0, 1, 0});
    
    vector_float4 c = color_at(w, r);
    
    // Should be a lit color, not black
    XCTAssertGreaterThan(c.x + c.y + c.z, 0.1);
}

// Chapter 7 - Finding the hit with all positive t
- (void)testHitWithAllPositiveT {
    Intersection xs[4];
    xs[0].t = 1.0;
    xs[0].objectId = 1;
    xs[1].t = 2.0;
    xs[1].objectId = 1;
    xs[2].t = 3.0;
    xs[2].objectId = 2;
    xs[3].t = 4.0;
    xs[3].objectId = 2;
    
    Intersection hit = hit_from_array(xs, 4);
    
    XCTAssertEqualWithAccuracy(hit.t, 1.0, 0.0001);
    XCTAssertEqual(hit.objectId, 1);
}

// Chapter 7 - Finding the hit with some negative t
- (void)testHitWithSomeNegativeT {
    Intersection xs[4];
    xs[0].t = -1.0;
    xs[0].objectId = 1;
    xs[1].t = 2.0;
    xs[1].objectId = 1;
    xs[2].t = -3.0;
    xs[2].objectId = 2;
    xs[3].t = 4.0;
    xs[3].objectId = 2;
    
    Intersection hit = hit_from_array(xs, 4);
    
    // Should find the smallest positive t
    XCTAssertEqualWithAccuracy(hit.t, 2.0, 0.0001);
    XCTAssertEqual(hit.objectId, 1);
}

// Chapter 7 - Finding the hit with all negative t
- (void)testHitWithAllNegativeT {
    Intersection xs[4];
    xs[0].t = -1.0;
    xs[0].objectId = 1;
    xs[1].t = -2.0;
    xs[1].objectId = 1;
    xs[2].t = -3.0;
    xs[2].objectId = 2;
    xs[3].t = -4.0;
    xs[3].objectId = 2;
    
    Intersection hit = hit_from_array(xs, 4);
    
    // Should indicate no hit
    XCTAssertEqual(hit.objectId, -1);
}

// Chapter 7 - Finding the hit from an unsorted array
- (void)testHitFromUnsortedArray {
    Intersection xs[4];
    xs[0].t = 5.0;
    xs[0].objectId = 1;
    xs[1].t = 3.0;
    xs[1].objectId = 1;
    xs[2].t = 7.0;
    xs[2].objectId = 2;
    xs[3].t = 1.0;
    xs[3].objectId = 2;
    
    Intersection hit = hit_from_array(xs, 4);
    
    // Should find the smallest positive t (1.0)
    XCTAssertEqualWithAccuracy(hit.t, 1.0, 0.0001);
    XCTAssertEqual(hit.objectId, 2);
}

// Chapter 7 - Adding objects to world
- (void)testAddingObjectsToWorld {
    World w = world_create_empty();
    
    Sphere s = sphere_create(1);
    world_add_sphere(&w, s);
    
    XCTAssertEqual(w.sphere_count, 1);
    XCTAssertEqual(w.spheres[0].id, 1);
}

@end
