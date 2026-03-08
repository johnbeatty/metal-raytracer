#import <XCTest/XCTest.h>
#import "SharedTypes.h"

@interface PlaneTests : XCTestCase
@end

@implementation PlaneTests

// Helper to check if two vectors are approximately equal
- (BOOL)vector:(vector_float4)a equalsVector:(vector_float4)b tolerance:(float)tol {
    return simd_all(simd_abs(a - b) < tol);
}

// Chapter 9 - The normal of a plane is constant everywhere
- (void)testNormalOfPlaneIsConstant {
    Plane p = plane_create(1);
    
    vector_float4 n1 = plane_normal_at(p, (vector_float4){0, 0, 0, 1});
    vector_float4 n2 = plane_normal_at(p, (vector_float4){10, 0, -10, 1});
    vector_float4 n3 = plane_normal_at(p, (vector_float4){-5, 0, 150, 1});
    
    vector_float4 expected = {0, 1, 0, 0};
    
    XCTAssertTrue([self vector:n1 equalsVector:expected tolerance:0.0001]);
    XCTAssertTrue([self vector:n2 equalsVector:expected tolerance:0.0001]);
    XCTAssertTrue([self vector:n3 equalsVector:expected tolerance:0.0001]);
}

// Chapter 9 - Intersect with a ray parallel to the plane
- (void)testIntersectRayParallelToPlane {
    Plane p = plane_create(1);
    Ray r = ray_create((vector_float4){0, 10, 0, 1}, (vector_float4){0, 0, 1, 0});
    
    float t;
    int hit = intersect_plane(p, r, &t);
    
    XCTAssertEqual(hit, 0);  // No intersection
}

// Chapter 9 - Intersect with a coplanar ray
- (void)testIntersectCoplanarRay {
    Plane p = plane_create(1);
    Ray r = ray_create((vector_float4){0, 0, 0, 1}, (vector_float4){0, 0, 1, 0});
    
    float t;
    int hit = intersect_plane(p, r, &t);
    
    XCTAssertEqual(hit, 0);  // No intersection (parallel)
}

// Chapter 9 - A ray intersecting a plane from above
- (void)testRayIntersectingPlaneFromAbove {
    Plane p = plane_create(1);
    Ray r = ray_create((vector_float4){0, 1, 0, 1}, (vector_float4){0, -1, 0, 0});
    
    float t;
    int hit = intersect_plane(p, r, &t);
    
    XCTAssertEqual(hit, 1);
    XCTAssertEqualWithAccuracy(t, 1.0, 0.0001);
}

// Chapter 9 - A ray intersecting a plane from below
- (void)testRayIntersectingPlaneFromBelow {
    Plane p = plane_create(1);
    Ray r = ray_create((vector_float4){0, -1, 0, 1}, (vector_float4){0, 1, 0, 0});
    
    float t;
    int hit = intersect_plane(p, r, &t);
    
    XCTAssertEqual(hit, 1);
    XCTAssertEqualWithAccuracy(t, 1.0, 0.0001);
}

// Chapter 9 - Plane with transformation
- (void)testPlaneWithTransformation {
    Plane p = plane_create(1);
    Matrix4x4 trans = matrix_translation(0, 3, 0);
    plane_set_transform(&p, trans);
    
    Ray r = ray_create((vector_float4){0, 4, 0, 1}, (vector_float4){0, -1, 0, 0});
    
    float t;
    int hit = intersect_plane(p, r, &t);
    
    XCTAssertEqual(hit, 1);
    XCTAssertEqualWithAccuracy(t, 1.0, 0.0001);
}

// Chapter 9 - Normal after transformation
- (void)testNormalAfterTransformation {
    Plane p = plane_create(1);
    Matrix4x4 trans = matrix_rotation_z(M_PI_4);  // Rotate 45 degrees
    plane_set_transform(&p, trans);
    
    vector_float4 n = plane_normal_at(p, (vector_float4){0, 0, 0, 1});
    
    float v = sqrtf(2.0) / 2.0;
    vector_float4 expected = {-v, v, 0, 0};  // Normal rotated
    
    XCTAssertTrue([self vector:n equalsVector:expected tolerance:0.0001]);
}

// Chapter 9 - Creating a plane
- (void)testCreatingPlane {
    Plane p = plane_create(42);
    
    XCTAssertEqual(p.id, 42);
    
    // Should have identity transform
    Matrix4x4 identity = MATRIX4X4_IDENTITY;
    for (int i = 0; i < 4; i++) {
        XCTAssertTrue(simd_equal(p.transform.columns[i], identity.columns[i]));
        XCTAssertTrue(simd_equal(p.inverseTransform.columns[i], identity.columns[i]));
    }
}

@end
