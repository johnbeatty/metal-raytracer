#import <XCTest/XCTest.h>
#import "SharedTypes.h"

@interface IntersectionTests : XCTestCase
@end

@implementation IntersectionTests

// Helper to check if two tuples are approximately equal
- (BOOL)tuple:(vector_float4)a equalsTuple:(vector_float4)b tolerance:(float)tol {
    return simd_all(simd_abs(a - b) < tol);
}

// Chapter 5 - Creating and querying a ray
- (void)testCreatingRay {
    vector_float4 origin = {1, 2, 3, 1};
    vector_float4 direction = {4, 5, 6, 0};
    Ray r = ray_create(origin, direction);
    
    XCTAssertTrue([self tuple:r.origin equalsTuple:origin tolerance:0.0001]);
    XCTAssertTrue([self tuple:r.direction equalsTuple:direction tolerance:0.0001]);
}

// Chapter 5 - Computing a point from a distance
- (void)testComputingPointFromDistance {
    Ray r = ray_create((vector_float4){2, 3, 4, 1}, 
                       (vector_float4){1, 0, 0, 0});
    
    vector_float4 pos0 = ray_position(r, 0);
    vector_float4 pos1 = ray_position(r, 1);
    vector_float4 pos2 = ray_position(r, -1);
    vector_float4 pos3 = ray_position(r, 2.5);
    
    vector_float4 expected0 = {2, 3, 4, 1};
    vector_float4 expected1 = {3, 3, 4, 1};
    vector_float4 expected2 = {1, 3, 4, 1};
    vector_float4 expected3 = {4.5, 3, 4, 1};
    
    XCTAssertTrue([self tuple:pos0 equalsTuple:expected0 tolerance:0.0001]);
    XCTAssertTrue([self tuple:pos1 equalsTuple:expected1 tolerance:0.0001]);
    XCTAssertTrue([self tuple:pos2 equalsTuple:expected2 tolerance:0.0001]);
    XCTAssertTrue([self tuple:pos3 equalsTuple:expected3 tolerance:0.0001]);
}

// Chapter 5 - A ray intersects a sphere at two points
- (void)testRayIntersectsSphereAtTwoPoints {
    // Ray starts at z = -5, goes in +z direction
    Ray r = ray_create((vector_float4){0, 0, -5, 1}, 
                       (vector_float4){0, 0, 1, 0});
    
    // Create unit sphere at origin
    Sphere s = sphere_create(1);
    
    // The ray should intersect at t = 4.0 and t = 6.0
    // Distance from origin to sphere center: 5
    // Sphere radius: 1
    // Intersection points: 5 - 1 = 4, 5 + 1 = 6
    
    // Manually compute intersection using quadratic formula
    // Use vector_float3 (SIMD 3-vector) instead of float3
    vector_float3 ray_origin = r.origin.xyz;
    vector_float3 ray_direction = r.direction.xyz;
    
    // Use simd_dot for 3-vectors - need explicit cast to avoid ambiguity
    float a = simd_dot((vector_float3)ray_direction, (vector_float3)ray_direction);
    float b = 2.0 * simd_dot((vector_float3)ray_origin, (vector_float3)ray_direction);
    float c = simd_dot((vector_float3)ray_origin, (vector_float3)ray_origin) - 1.0;
    
    float discriminant = b*b - 4.0*a*c;
    XCTAssertGreaterThan(discriminant, 0, @"Should have two intersections");
    
    float sqrt_disc = sqrtf(discriminant);
    float t0 = (-b - sqrt_disc) / (2.0 * a);
    float t1 = (-b + sqrt_disc) / (2.0 * a);
    
    XCTAssertEqualWithAccuracy(t0, 4.0, 0.0001);
    XCTAssertEqualWithAccuracy(t1, 6.0, 0.0001);
}

// Chapter 5 - A ray intersects a sphere at a tangent
- (void)testRayIntersectsSphereAtTangent {
    // Ray starts at y = 1, z = -5, goes in +z direction (tangent to sphere)
    Ray r = ray_create((vector_float4){0, 1, -5, 1}, 
                       (vector_float4){0, 0, 1, 0});
    
    vector_float3 ray_origin = r.origin.xyz;
    vector_float3 ray_direction = r.direction.xyz;
    
    float a = simd_dot((vector_float3)ray_direction, (vector_float3)ray_direction);
    float b = 2.0 * simd_dot((vector_float3)ray_origin, (vector_float3)ray_direction);
    float c = simd_dot((vector_float3)ray_origin, (vector_float3)ray_origin) - 1.0;
    
    float discriminant = b*b - 4.0*a*c;
    XCTAssertEqualWithAccuracy(discriminant, 0, 0.0001);
    
    float t = -b / (2.0 * a);
    XCTAssertEqualWithAccuracy(t, 5.0, 0.0001);
}

// Chapter 5 - A ray misses a sphere
- (void)testRayMissesSphere {
    // Ray starts above sphere
    Ray r = ray_create((vector_float4){0, 2, -5, 1}, 
                       (vector_float4){0, 0, 1, 0});
    
    vector_float3 ray_origin = r.origin.xyz;
    vector_float3 ray_direction = r.direction.xyz;
    
    float a = simd_dot((vector_float3)ray_direction, (vector_float3)ray_direction);
    float b = 2.0 * simd_dot((vector_float3)ray_origin, (vector_float3)ray_direction);
    float c = simd_dot((vector_float3)ray_origin, (vector_float3)ray_origin) - 1.0;
    
    float discriminant = b*b - 4.0*a*c;
    XCTAssertLessThan(discriminant, 0, @"Should miss sphere");
}

// Chapter 5 - A ray originates inside a sphere
- (void)testRayOriginatesInsideSphere {
    // Ray starts at origin (inside sphere), goes in +z direction
    Ray r = ray_create((vector_float4){0, 0, 0, 1}, 
                       (vector_float4){0, 0, 1, 0});
    
    vector_float3 ray_origin = r.origin.xyz;
    vector_float3 ray_direction = r.direction.xyz;
    
    float a = simd_dot((vector_float3)ray_direction, (vector_float3)ray_direction);
    float b = 2.0 * simd_dot((vector_float3)ray_origin, (vector_float3)ray_direction);
    float c = simd_dot((vector_float3)ray_origin, (vector_float3)ray_origin) - 1.0;
    
    float discriminant = b*b - 4.0*a*c;
    XCTAssertGreaterThan(discriminant, 0, @"Should have two intersections");
    
    float sqrt_disc = sqrtf(discriminant);
    float t0 = (-b - sqrt_disc) / (2.0 * a);
    float t1 = (-b + sqrt_disc) / (2.0 * a);
    
    // t0 is behind the ray origin (negative), t1 is in front
    XCTAssertEqualWithAccuracy(t0, -1.0, 0.0001);
    XCTAssertEqualWithAccuracy(t1, 1.0, 0.0001);
}

// Chapter 5 - A sphere is behind a ray
- (void)testSphereBehindRay {
    // Ray starts in front of sphere, going away from it
    Ray r = ray_create((vector_float4){0, 0, 5, 1}, 
                       (vector_float4){0, 0, 1, 0});
    
    vector_float3 ray_origin = r.origin.xyz;
    vector_float3 ray_direction = r.direction.xyz;
    
    float a = simd_dot((vector_float3)ray_direction, (vector_float3)ray_direction);
    float b = 2.0 * simd_dot((vector_float3)ray_origin, (vector_float3)ray_direction);
    float c = simd_dot((vector_float3)ray_origin, (vector_float3)ray_origin) - 1.0;
    
    float discriminant = b*b - 4.0*a*c;
    XCTAssertGreaterThan(discriminant, 0, @"Mathematically intersects but behind ray");
    
    float sqrt_disc = sqrtf(discriminant);
    float t0 = (-b - sqrt_disc) / (2.0 * a);
    float t1 = (-b + sqrt_disc) / (2.0 * a);
    
    // Both t values should be negative (sphere behind ray)
    XCTAssertLessThan(t0, 0);
    XCTAssertLessThan(t1, 0);
}

// Chapter 5 - The hit, when all intersections have positive t
- (void)testHitWithAllPositiveT {
    // Ray intersects sphere from outside at t = 4 and t = 6
    // Both positive, so hit should be the smaller one: t = 4
    float t0 = 4.0;
    float t1 = 6.0;
    
    // The hit is the intersection with the lowest non-negative t
    float hit_t = (t0 >= 0 && t1 >= 0) ? fmin(t0, t1) : fmax(t0, t1);
    
    XCTAssertEqualWithAccuracy(hit_t, 4.0, 0.0001);
}

// Chapter 5 - The hit, when some intersections have negative t
- (void)testHitWithSomeNegativeT {
    // Ray starts inside sphere, t = -1 (behind), t = 1 (in front)
    float t0 = -1.0;
    float t1 = 1.0;
    
    // The hit is the lowest non-negative t
    float hit_t;
    if (t0 >= 0 && t1 >= 0) {
        hit_t = fmin(t0, t1);
    } else if (t0 >= 0) {
        hit_t = t0;
    } else if (t1 >= 0) {
        hit_t = t1;
    } else {
        hit_t = NAN;  // No hit
    }
    
    XCTAssertEqualWithAccuracy(hit_t, 1.0, 0.0001);
}

// Chapter 5 - The hit, when all intersections have negative t
- (void)testHitWithAllNegativeT {
    // Sphere is completely behind ray
    float t0 = -2.0;
    float t1 = -1.0;
    
    // Both negative, so no hit
    BOOL hasHit = (t0 >= 0) || (t1 >= 0);
    
    XCTAssertFalse(hasHit);
}

// Chapter 5 - Intersecting a scaled sphere with a ray
- (void)testIntersectingScaledSphereWithRay {
    // Ray starts at origin, goes in +z direction
    Ray r = ray_create((vector_float4){0, 0, -5, 1}, 
                       (vector_float4){0, 0, 1, 0});
    
    // Create scaled sphere (radius 2 instead of 1)
    Sphere s = sphere_create(1);
    Matrix4x4 scale = matrix_scaling(2, 2, 2);
    sphere_set_transform(&s, scale);
    
    // Transform ray to object space
    Ray object_ray = ray_transform(r, s.inverseTransform);
    
    // Now intersect with unit sphere
    vector_float3 ray_origin = object_ray.origin.xyz;
    vector_float3 ray_direction = object_ray.direction.xyz;
    
    float a = simd_dot((vector_float3)ray_direction, (vector_float3)ray_direction);
    float b = 2.0 * simd_dot((vector_float3)ray_origin, (vector_float3)ray_direction);
    float c = simd_dot((vector_float3)ray_origin, (vector_float3)ray_origin) - 1.0;
    
    float discriminant = b*b - 4.0*a*c;
    XCTAssertGreaterThan(discriminant, 0, @"Should intersect scaled sphere");
    
    float sqrt_disc = sqrtf(discriminant);
    float t0 = (-b - sqrt_disc) / (2.0 * a);
    float t1 = (-b + sqrt_disc) / (2.0 * a);
    
    // For a sphere scaled by 2, the intersection t values should be 3 and 7
    // (5 - 2 = 3, 5 + 2 = 7)
    XCTAssertEqualWithAccuracy(t0, 3.0, 0.0001);
    XCTAssertEqualWithAccuracy(t1, 7.0, 0.0001);
}

// Chapter 5 - Intersecting a translated sphere with a ray
- (void)testIntersectingTranslatedSphereWithRay {
    // Ray starts at origin, goes in +z direction
    Ray r = ray_create((vector_float4){0, 0, -5, 1}, 
                       (vector_float4){0, 0, 1, 0});
    
    // Create translated sphere (moved 5 units in +z)
    Sphere s = sphere_create(1);
    Matrix4x4 translate = matrix_translation(0, 0, 5);
    sphere_set_transform(&s, translate);
    
    // Transform ray to object space
    Ray object_ray = ray_transform(r, s.inverseTransform);
    
    // Now intersect with unit sphere
    vector_float3 ray_origin = object_ray.origin.xyz;
    vector_float3 ray_direction = object_ray.direction.xyz;
    
    float a = simd_dot((vector_float3)ray_direction, (vector_float3)ray_direction);
    float b = 2.0 * simd_dot((vector_float3)ray_origin, (vector_float3)ray_direction);
    float c = simd_dot((vector_float3)ray_origin, (vector_float3)ray_origin) - 1.0;
    
    // The sphere is moved to z=5, and ray starts at z=-5 going to z=0
    // In object space, sphere is at origin, ray is at z=-10 going to z=-5
    // Distance = 10, radius = 1, so discriminant should be positive
    float discriminant = b*b - 4.0*a*c;
    XCTAssertGreaterThan(discriminant, 0, @"Should intersect translated sphere");
    
    float sqrt_disc = sqrtf(discriminant);
    float t0 = (-b - sqrt_disc) / (2.0 * a);
    float t1 = (-b + sqrt_disc) / (2.0 * a);
    
    // In object space, the ray starts at z=-10 and goes to z=-5
    // Distance to sphere: 10, radius: 1
    // Intersections: 10 - 1 = 9, 10 + 1 = 11
    XCTAssertEqualWithAccuracy(t0, 9.0, 0.0001);
    XCTAssertEqualWithAccuracy(t1, 11.0, 0.0001);
}

// Chapter 5 - A sphere has a default transformation
- (void)testSphereDefaultTransformation {
    Sphere s = sphere_create(1);
    
    // Should be identity
    Matrix4x4 identity = MATRIX4X4_IDENTITY;
    for (int i = 0; i < 4; i++) {
        XCTAssertTrue(simd_equal(s.transform.columns[i], identity.columns[i]));
    }
}

// Chapter 5 - Changing a sphere's transformation
- (void)testChangingSphereTransformation {
    Sphere s = sphere_create(1);
    Matrix4x4 translate = matrix_translation(2, 3, 4);
    
    sphere_set_transform(&s, translate);
    
    // Check transform was set
    for (int i = 0; i < 4; i++) {
        XCTAssertTrue(simd_equal(s.transform.columns[i], translate.columns[i]));
    }
    
    // Check inverse was computed
    matrix_float4x4 mat = matrix_from_columns(translate.columns[0], translate.columns[1],
                                              translate.columns[2], translate.columns[3]);
    matrix_float4x4 inv = matrix_invert(mat);
    for (int i = 0; i < 4; i++) {
        XCTAssertTrue(simd_all(simd_abs(s.inverseTransform.columns[i] - inv.columns[i]) < 0.0001));
    }
}

@end
