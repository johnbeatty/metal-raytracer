#import <XCTest/XCTest.h>
#import "SharedTypes.h"

@interface CubeTests : XCTestCase
@end

@implementation CubeTests

// Helper to check if two values are approximately equal
- (BOOL)float:(float)a equals:(float)b tolerance:(float)tol {
    return fabsf(a - b) < tol;
}

// Helper to check if two vectors are approximately equal
- (BOOL)vector:(vector_float4)a equalsVector:(vector_float4)b tolerance:(float)tol {
    return simd_all(simd_abs(a - b) < tol);
}

// MARK: - Ray-Cube Intersection Tests

// Chapter 12 - A ray intersects a cube
- (void)testRayIntersectsCube {
    Cube c = cube_create(1);
    
    // Ray from outside, pointing at center
    Ray ray;
    ray.origin = (vector_float4){5, 0.5, 0, 1};
    ray.direction = (vector_float4){-1, 0, 0, 0};
    
    float t0, t1;
    int count = intersect_cube(c, ray, &t0, &t1);
    
    XCTAssertEqual(count, 2);
    XCTAssertTrue([self float:t0 equals:4 tolerance:0.0001]);
    XCTAssertTrue([self float:t1 equals:6 tolerance:0.0001]);
}

// Chapter 12 - A ray misses a cube
- (void)testRayMissesCube {
    Cube c = cube_create(1);
    
    // Ray parallel to cube, not intersecting
    Ray ray;
    ray.origin = (vector_float4){-2, 0, 0, 1};
    ray.direction = (vector_float4){0, 1, 0, 0};
    
    float t0, t1;
    int count = intersect_cube(c, ray, &t0, &t1);
    
    XCTAssertEqual(count, 0);
}

// Chapter 12 - A ray misses a cube from inside
- (void)testRayInsideCubeMisses {
    Cube c = cube_create(1);
    
    // Ray starting inside, going outward
    Ray ray;
    ray.origin = (vector_float4){0, 0, 0, 1};
    ray.direction = (vector_float4){1, 0, 0, 0};
    
    float t0, t1;
    int count = intersect_cube(c, ray, &t0, &t1);
    
    // Should hit one face at t=1
    XCTAssertGreaterThanOrEqual(count, 1);
}

// Chapter 12 - Intersection from +x direction
- (void)testIntersectionFromPositiveX {
    Cube c = cube_create(1);
    
    Ray ray;
    ray.origin = (vector_float4){5, 0.5, 0, 1};
    ray.direction = (vector_float4){-1, 0, 0, 0};
    
    float t0, t1;
    int count = intersect_cube(c, ray, &t0, &t1);
    
    XCTAssertEqual(count, 2);
    XCTAssertTrue([self float:t0 equals:4 tolerance:0.0001]);
    XCTAssertTrue([self float:t1 equals:6 tolerance:0.0001]);
}

// Chapter 12 - Intersection from -x direction
- (void)testIntersectionFromNegativeX {
    Cube c = cube_create(1);
    
    Ray ray;
    ray.origin = (vector_float4){-5, 0.5, 0, 1};
    ray.direction = (vector_float4){1, 0, 0, 0};
    
    float t0, t1;
    int count = intersect_cube(c, ray, &t0, &t1);
    
    XCTAssertEqual(count, 2);
    XCTAssertTrue([self float:t0 equals:4 tolerance:0.0001]);
    XCTAssertTrue([self float:t1 equals:6 tolerance:0.0001]);
}

// Chapter 12 - Intersection from +y direction
- (void)testIntersectionFromPositiveY {
    Cube c = cube_create(1);
    
    Ray ray;
    ray.origin = (vector_float4){0.5, 5, 0, 1};
    ray.direction = (vector_float4){0, -1, 0, 0};
    
    float t0, t1;
    int count = intersect_cube(c, ray, &t0, &t1);
    
    XCTAssertEqual(count, 2);
    XCTAssertTrue([self float:t0 equals:4 tolerance:0.0001]);
    XCTAssertTrue([self float:t1 equals:6 tolerance:0.0001]);
}

// Chapter 12 - Intersection from -y direction
- (void)testIntersectionFromNegativeY {
    Cube c = cube_create(1);
    
    Ray ray;
    ray.origin = (vector_float4){0.5, -5, 0, 1};
    ray.direction = (vector_float4){0, 1, 0, 0};
    
    float t0, t1;
    int count = intersect_cube(c, ray, &t0, &t1);
    
    XCTAssertEqual(count, 2);
    XCTAssertTrue([self float:t0 equals:4 tolerance:0.0001]);
    XCTAssertTrue([self float:t1 equals:6 tolerance:0.0001]);
}

// Chapter 12 - Intersection from +z direction
- (void)testIntersectionFromPositiveZ {
    Cube c = cube_create(1);
    
    Ray ray;
    ray.origin = (vector_float4){0.5, 0, 5, 1};
    ray.direction = (vector_float4){0, 0, -1, 0};
    
    float t0, t1;
    int count = intersect_cube(c, ray, &t0, &t1);
    
    XCTAssertEqual(count, 2);
    XCTAssertTrue([self float:t0 equals:4 tolerance:0.0001]);
    XCTAssertTrue([self float:t1 equals:6 tolerance:0.0001]);
}

// Chapter 12 - Intersection from -z direction
- (void)testIntersectionFromNegativeZ {
    Cube c = cube_create(1);
    
    Ray ray;
    ray.origin = (vector_float4){0.5, 0, -5, 1};
    ray.direction = (vector_float4){0, 0, 1, 0};
    
    float t0, t1;
    int count = intersect_cube(c, ray, &t0, &t1);
    
    XCTAssertEqual(count, 2);
    XCTAssertTrue([self float:t0 equals:4 tolerance:0.0001]);
    XCTAssertTrue([self float:t1 equals:6 tolerance:0.0001]);
}

// Chapter 12 - Ray inside cube
- (void)testRayInsideCube {
    Cube c = cube_create(1);
    
    Ray ray;
    ray.origin = (vector_float4){0, 0.5, 0, 1};
    ray.direction = (vector_float4){1, 0, 0, 0};
    
    float t0, t1;
    int count = intersect_cube(c, ray, &t0, &t1);
    
    // Inside, should only hit exit face
    XCTAssertGreaterThanOrEqual(count, 1);
    XCTAssertTrue([self float:t0 equals:1 tolerance:0.0001]);
}

// Chapter 12 - Ray misses cube (corner)
- (void)testRayMissesCubeCorner {
    Cube c = cube_create(1);
    
    Ray ray;
    ray.origin = (vector_float4){-2, 2, 2, 1};
    ray.direction = (vector_float4){0, -1, -1, 0};
    ray.direction = simd_normalize(ray.direction);
    
    float t0, t1;
    int count = intersect_cube(c, ray, &t0, &t1);
    
    XCTAssertEqual(count, 0);
}

// MARK: - Cube Normal Tests

// Chapter 12 - The normal on the surface of a cube
- (void)testCubeNormalPositiveX {
    Cube c = cube_create(1);
    vector_float4 p = {1, 0.5, -0.5, 1};
    vector_float4 normal = cube_normal_at(c, p);
    vector_float4 expected = {1, 0, 0, 0};
    
    XCTAssertTrue([self vector:normal equalsVector:expected tolerance:0.0001]);
}

- (void)testCubeNormalNegativeX {
    Cube c = cube_create(1);
    vector_float4 p = {-1, -0.2, 0.3, 1};
    vector_float4 normal = cube_normal_at(c, p);
    vector_float4 expected = {-1, 0, 0, 0};
    
    XCTAssertTrue([self vector:normal equalsVector:expected tolerance:0.0001]);
}

- (void)testCubeNormalPositiveY {
    Cube c = cube_create(1);
    vector_float4 p = {-0.4, 1, -0.1, 1};
    vector_float4 normal = cube_normal_at(c, p);
    vector_float4 expected = {0, 1, 0, 0};
    
    XCTAssertTrue([self vector:normal equalsVector:expected tolerance:0.0001]);
}

- (void)testCubeNormalNegativeY {
    Cube c = cube_create(1);
    vector_float4 p = {0.3, -1, -0.7, 1};
    vector_float4 normal = cube_normal_at(c, p);
    vector_float4 expected = {0, -1, 0, 0};
    
    XCTAssertTrue([self vector:normal equalsVector:expected tolerance:0.0001]);
}

- (void)testCubeNormalPositiveZ {
    Cube c = cube_create(1);
    vector_float4 p = {-0.6, 0.3, 1, 1};
    vector_float4 normal = cube_normal_at(c, p);
    vector_float4 expected = {0, 0, 1, 0};
    
    XCTAssertTrue([self vector:normal equalsVector:expected tolerance:0.0001]);
}

- (void)testCubeNormalNegativeZ {
    Cube c = cube_create(1);
    vector_float4 p = {0.4, 0.4, -1, 1};
    vector_float4 normal = cube_normal_at(c, p);
    vector_float4 expected = {0, 0, -1, 0};
    
    XCTAssertTrue([self vector:normal equalsVector:expected tolerance:0.0001]);
}

- (void)testCubeNormalOnSurface {
    Cube c = cube_create(1);
    vector_float4 p = {-1, 1, -1, 1};
    vector_float4 normal = cube_normal_at(c, p);
    // At a corner, should pick the face with largest absolute value
    // All are 1, so it could be any. The implementation picks x first.
    vector_float4 expected = {-1, 0, 0, 0};
    
    XCTAssertTrue([self vector:normal equalsVector:expected tolerance:0.0001]);
}

// MARK: - Cube Creation Tests

// Chapter 12 - Creating a cube
- (void)testCreatingCube {
    Cube c = cube_create(1);
    
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

// Chapter 12 - Setting cube transform
- (void)testSettingCubeTransform {
    Cube c = cube_create(1);
    
    Matrix4x4 trans = matrix_translation(2, 3, 4);
    cube_set_transform(&c, trans);
    
    vector_float4 expected = {2, 3, 4, 1};
    XCTAssertTrue([self vector:c.transform.columns[3] equalsVector:expected tolerance:0.0001]);
}

// Chapter 12 - Cube has default material
- (void)testCubeDefaultMaterial {
    Cube c = cube_create(1);
    
    // DEFAULT_MATERIAL values: ambient=0.1, diffuse=0.9, specular=0.9
    XCTAssertEqualWithAccuracy(c.material.ambient, 0.1f, 0.0001);
    XCTAssertEqualWithAccuracy(c.material.diffuse, 0.9f, 0.0001);
    XCTAssertEqualWithAccuracy(c.material.specular, 0.9f, 0.0001);
}

// Chapter 12 - Setting cube material
- (void)testSettingCubeMaterial {
    Cube c = cube_create(1);
    
    c.material.color = (vector_float4){1, 0, 0, 1};
    c.material.ambient = 0.2;
    
    vector_float4 expected_color = {1, 0, 0, 1};
    XCTAssertTrue([self vector:c.material.color equalsVector:expected_color tolerance:0.0001]);
    XCTAssertEqualWithAccuracy(c.material.ambient, 0.2, 0.0001);
}

// Chapter 12 - Ray intersects translated cube
- (void)testRayIntersectsTranslatedCube {
    Cube c = cube_create(1);
    Matrix4x4 trans = matrix_translation(2, 0, 0);
    cube_set_transform(&c, trans);
    
    Ray ray;
    ray.origin = (vector_float4){7, 0.5, 0, 1};
    ray.direction = (vector_float4){-1, 0, 0, 0};
    
    float t0, t1;
    int count = intersect_cube(c, ray, &t0, &t1);
    
    // Ray should hit the translated cube at x=1, x=3
    XCTAssertEqual(count, 2);
}

// Chapter 12 - Ray intersects scaled cube
- (void)testRayIntersectsScaledCube {
    Cube c = cube_create(1);
    Matrix4x4 scale = matrix_scaling(2, 2, 2);
    cube_set_transform(&c, scale);
    
    Ray ray;
    ray.origin = (vector_float4){5, 0, 0, 1};
    ray.direction = (vector_float4){-1, 0, 0, 0};
    
    float t0, t1;
    int count = intersect_cube(c, ray, &t0, &t1);
    
    // Scaled cube goes from -2 to 2, so ray hits at t=3 and t=7
    XCTAssertEqual(count, 2);
}

@end
