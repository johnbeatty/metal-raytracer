#import <XCTest/XCTest.h>
#import "SharedTypes.h"

@interface CylinderTests : XCTestCase
@end

@implementation CylinderTests

// Helper to check if two values are approximately equal
- (BOOL)float:(float)a equals:(float)b tolerance:(float)tol {
    return fabsf(a - b) < tol;
}

// Helper to check if two vectors are approximately equal
- (BOOL)vector:(vector_float4)a equalsVector:(vector_float4)b tolerance:(float)tol {
    return simd_all(simd_abs(a - b) < tol);
}

// MARK: - Cylinder Creation Tests

// Chapter 13 - Creating a cylinder
- (void)testCreatingCylinder {
    Cylinder c = cylinder_create(1);
    
    XCTAssertEqual(c.id, 1);
    
    vector_float4 col0 = {1, 0, 0, 0};
    XCTAssertTrue([self vector:c.transform.columns[0] equalsVector:col0 tolerance:0.0001]);
    
    vector_float4 col1 = {0, 1, 0, 0};
    XCTAssertTrue([self vector:c.transform.columns[1] equalsVector:col1 tolerance:0.0001]);
    
    vector_float4 col2 = {0, 0, 1, 0};
    XCTAssertTrue([self vector:c.transform.columns[2] equalsVector:col2 tolerance:0.0001]);
    
    vector_float4 col3 = {0, 0, 0, 1};
    XCTAssertTrue([self vector:c.transform.columns[3] equalsVector:col3 tolerance:0.0001]);
}

// Chapter 13 - Default cylinder is infinite
- (void)testDefaultCylinderIsInfinite {
    Cylinder c = cylinder_create(1);
    
    XCTAssertEqual(c.minimum, -INFINITY);
    XCTAssertEqual(c.maximum, INFINITY);
    XCTAssertFalse(c.closed);
}

// Chapter 13 - Setting cylinder transform
- (void)testSettingCylinderTransform {
    Cylinder c = cylinder_create(1);
    
    Matrix4x4 trans = matrix_translation(2, 3, 4);
    cylinder_set_transform(&c, trans);
    
    vector_float4 expected = {2, 3, 4, 1};
    XCTAssertTrue([self vector:c.transform.columns[3] equalsVector:expected tolerance:0.0001]);
}

// MARK: - Ray-Cylinder Intersection Tests

// Chapter 13 - A ray misses a cylinder
- (void)testRayMissesCylinder {
    Cylinder c = cylinder_create(1);
    
    // Ray passing by the cylinder
    Ray ray;
    ray.origin = (vector_float4){1, 0, 0, 1};
    ray.direction = (vector_float4){0, 1, 0, 0};
    
    float t0, t1;
    int count = intersect_cylinder(c, ray, &t0, &t1);
    
    XCTAssertEqual(count, 0);
}

// Chapter 13 - A ray intersects a cylinder
- (void)testRayIntersectsCylinder {
    Cylinder c = cylinder_create(1);
    
    Ray ray;
    ray.origin = (vector_float4){0, 0, -5, 1};
    ray.direction = (vector_float4){0, 0, 1, 0};
    
    float t0, t1;
    int count = intersect_cylinder(c, ray, &t0, &t1);
    
    XCTAssertEqual(count, 2);
}

// Chapter 13 - Ray misses when passing by
- (void)testRayPassesByCylinder {
    Cylinder c = cylinder_create(1);
    
    Ray ray;
    ray.origin = (vector_float4){2, 0, 0, 1};
    ray.direction = (vector_float4){0, 1, 0, 0};
    
    float t0, t1;
    int count = intersect_cylinder(c, ray, &t0, &t1);
    
    XCTAssertEqual(count, 0);
}

// Chapter 13 - Ray starting outside and going through
- (void)testRayFromOutsideThroughCylinder {
    Cylinder c = cylinder_create(1);
    
    Ray ray;
    ray.origin = (vector_float4){0, 0, -5, 1};
    ray.direction = (vector_float4){0, 0, 1, 0};
    
    float t0, t1;
    int count = intersect_cylinder(c, ray, &t0, &t1);
    
    XCTAssertEqual(count, 2);
    XCTAssertTrue([self float:t0 equals:4 tolerance:0.0001]);
    XCTAssertTrue([self float:t1 equals:6 tolerance:0.0001]);
}

// Chapter 13 - Ray starting inside cylinder
- (void)testRayFromInsideCylinder {
    Cylinder c = cylinder_create(1);
    
    Ray ray;
    ray.origin = (vector_float4){0, 0, 0, 1};  // Inside
    ray.direction = (vector_float4){0, 0, 1, 0};
    
    float t0, t1;
    int count = intersect_cylinder(c, ray, &t0, &t1);
    
    // Should hit exit at t=1
    XCTAssertGreaterThanOrEqual(count, 1);
    XCTAssertTrue([self float:t0 equals:1 tolerance:0.0001]);
}

// MARK: - Truncated Cylinder Tests

// Chapter 13 - Truncated cylinder with minimum and maximum
- (void)testTruncatedCylinder {
    Cylinder c = cylinder_create(1);
    c.minimum = 1.0f;
    c.maximum = 2.0f;
    
    // Ray going through the truncated region
    Ray ray;
    ray.origin = (vector_float4){0, 1.5, -5, 1};
    ray.direction = (vector_float4){0, 0, 1, 0};
    
    float t0, t1;
    int count = intersect_cylinder(c, ray, &t0, &t1);
    
    XCTAssertEqual(count, 2);
}

// Chapter 13 - Ray misses truncated cylinder (below minimum)
- (void)testRayMissesTruncatedCylinderBelow {
    Cylinder c = cylinder_create(1);
    c.minimum = 1.0f;
    c.maximum = 2.0f;
    
    Ray ray;
    ray.origin = (vector_float4){0, 0.5, -5, 1};
    ray.direction = (vector_float4){0, 0, 1, 0};
    
    float t0, t1;
    int count = intersect_cylinder(c, ray, &t0, &t1);
    
    XCTAssertEqual(count, 0);
}

// Chapter 13 - Ray misses truncated cylinder (above maximum)
- (void)testRayMissesTruncatedCylinderAbove {
    Cylinder c = cylinder_create(1);
    c.minimum = 1.0f;
    c.maximum = 2.0f;
    
    Ray ray;
    ray.origin = (vector_float4){0, 2.5, -5, 1};
    ray.direction = (vector_float4){0, 0, 1, 0};
    
    float t0, t1;
    int count = intersect_cylinder(c, ray, &t0, &t1);
    
    XCTAssertEqual(count, 0);
}

// MARK: - Cylinder Normal Tests

// Chapter 13 - Normal on cylinder surface
- (void)testNormalOnCylinderSurface {
    Cylinder c = cylinder_create(1);
    
    vector_float4 p = {1, 0, 0, 1};
    vector_float4 normal = cylinder_normal_at(c, p);
    vector_float4 expected = {1, 0, 0, 0};
    
    XCTAssertTrue([self vector:normal equalsVector:expected tolerance:0.0001]);
}

// Chapter 13 - Normal on cylinder at different points
- (void)testNormalOnCylinderAtDifferentPoints {
    Cylinder c = cylinder_create(1);
    
    // Point at (0.5, 1, 0) on surface
    vector_float4 p1 = {0.5, 1, 0, 1};
    // Distance from axis is sqrt(0.5^2) = 0.5, which is within unit cylinder
    // Actually, this point is INSIDE the cylinder, not on surface
    // Let's use a point actually on the surface
    
    // Point at (1, 0, 0) - on the surface
    vector_float4 p2 = {1, 0, 0, 1};
    vector_float4 normal2 = cylinder_normal_at(c, p2);
    vector_float4 expected2 = {1, 0, 0, 0};
    XCTAssertTrue([self vector:normal2 equalsVector:expected2 tolerance:0.0001]);
    
    // Point at (0, 0, 1) - on the surface
    vector_float4 p3 = {0, 0, 1, 1};
    vector_float4 normal3 = cylinder_normal_at(c, p3);
    vector_float4 expected3 = {0, 0, 1, 0};
    XCTAssertTrue([self vector:normal3 equalsVector:expected3 tolerance:0.0001]);
}

// Chapter 13 - Normal varies on cylinder surface
- (void)testNormalVariesOnCylinder {
    Cylinder c = cylinder_create(1);
    
    // Points at different angles around the cylinder
    vector_float4 p1 = {1, 0, 0, 1};  // Angle 0
    vector_float4 n1 = cylinder_normal_at(c, p1);
    vector_float4 expected1 = {1, 0, 0, 0};
    XCTAssertTrue([self vector:n1 equalsVector:expected1 tolerance:0.0001]);
    
    // Point at (0, 0, 1) - angle 90 degrees
    vector_float4 p2 = {0, 0, 1, 1};
    vector_float4 n2 = cylinder_normal_at(c, p2);
    vector_float4 expected2 = {0, 0, 1, 0};
    XCTAssertTrue([self vector:n2 equalsVector:expected2 tolerance:0.0001]);
    
    // Point at (-1, 0, 0) - angle 180 degrees
    vector_float4 p3 = {-1, 0, 0, 1};
    vector_float4 n3 = cylinder_normal_at(c, p3);
    vector_float4 expected3 = {-1, 0, 0, 0};
    XCTAssertTrue([self vector:n3 equalsVector:expected3 tolerance:0.0001]);
    
    // Point at (0, 0, -1) - angle 270 degrees
    vector_float4 p4 = {0, 0, -1, 1};
    vector_float4 n4 = cylinder_normal_at(c, p4);
    vector_float4 expected4 = {0, 0, -1, 0};
    XCTAssertTrue([self vector:n4 equalsVector:expected4 tolerance:0.0001]);
}

// MARK: - Cylinder Material Tests

// Chapter 13 - Cylinder has default material
- (void)testCylinderDefaultMaterial {
    Cylinder c = cylinder_create(1);
    
    XCTAssertEqualWithAccuracy(c.material.ambient, 0.1f, 0.0001);
    XCTAssertEqualWithAccuracy(c.material.diffuse, 0.9f, 0.0001);
    XCTAssertEqualWithAccuracy(c.material.specular, 0.9f, 0.0001);
}

// Chapter 13 - Setting cylinder material
- (void)testSettingCylinderMaterial {
    Cylinder c = cylinder_create(1);
    
    c.material.color = (vector_float4){1, 0, 0, 1};
    c.material.ambient = 0.2;
    
    vector_float4 expected_color = {1, 0, 0, 1};
    XCTAssertTrue([self vector:c.material.color equalsVector:expected_color tolerance:0.0001]);
    XCTAssertEqualWithAccuracy(c.material.ambient, 0.2, 0.0001);
}

// MARK: - Scaled and Transformed Cylinder Tests

// Chapter 13 - Ray intersects scaled cylinder
- (void)testRayIntersectsScaledCylinder {
    Cylinder c = cylinder_create(1);
    Matrix4x4 scale = matrix_scaling(2, 2, 2);
    cylinder_set_transform(&c, scale);
    
    // After scaling by 2, the cylinder has radius 2
    // Ray at distance 1.5 from center should hit
    Ray ray;
    ray.origin = (vector_float4){1.5, 0, -5, 1};
    ray.direction = (vector_float4){0, 0, 1, 0};
    
    float t0, t1;
    int count = intersect_cylinder(c, ray, &t0, &t1);
    
    XCTAssertEqual(count, 2);
}

// Chapter 13 - Ray intersects translated cylinder
- (void)testRayIntersectsTranslatedCylinder {
    Cylinder c = cylinder_create(1);
    Matrix4x4 trans = matrix_translation(2, 0, 0);
    cylinder_set_transform(&c, trans);
    
    // Cylinder center is now at (2, 0, 0)
    Ray ray;
    ray.origin = (vector_float4){2, 0, -5, 1};
    ray.direction = (vector_float4){0, 0, 1, 0};
    
    float t0, t1;
    int count = intersect_cylinder(c, ray, &t0, &t1);
    
    XCTAssertEqual(count, 2);
}

@end
