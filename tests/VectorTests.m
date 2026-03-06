#import <XCTest/XCTest.h>
#import "SharedTypes.h"

@interface VectorTests : XCTestCase
@end

@implementation VectorTests

// Chapter 1 - Vector operations

- (void)testVectorMagnitudeX {
    // Computing the magnitude of vector (1, 0, 0)
    Tuple v = { .components = {1.0, 0.0, 0.0, 0.0} };
    float mag = simd_length(v.components.xyz);
    XCTAssertEqual(mag, 1.0, @"Magnitude of (1,0,0) should be 1");
}

- (void)testVectorMagnitudeY {
    // Computing the magnitude of vector (0, 1, 0)
    Tuple v = { .components = {0.0, 1.0, 0.0, 0.0} };
    float mag = simd_length(v.components.xyz);
    XCTAssertEqual(mag, 1.0, @"Magnitude of (0,1,0) should be 1");
}

- (void)testVectorMagnitudeZ {
    // Computing the magnitude of vector (0, 0, 1)
    Tuple v = { .components = {0.0, 0.0, 1.0, 0.0} };
    float mag = simd_length(v.components.xyz);
    XCTAssertEqual(mag, 1.0, @"Magnitude of (0,0,1) should be 1");
}

- (void)testVectorMagnitudePositive {
    // Computing the magnitude of vector (1, 2, 3)
    Tuple v = { .components = {1.0, 2.0, 3.0, 0.0} };
    float mag = simd_length(v.components.xyz);
    XCTAssertEqualWithAccuracy(mag, sqrtf(14.0), 0.0001, @"Magnitude of (1,2,3) should be sqrt(14)");
}

- (void)testVectorMagnitudeNegative {
    // Computing the magnitude of vector (-1, -2, -3)
    Tuple v = { .components = {-1.0, -2.0, -3.0, 0.0} };
    float mag = simd_length(v.components.xyz);
    XCTAssertEqualWithAccuracy(mag, sqrtf(14.0), 0.0001, @"Magnitude of (-1,-2,-3) should be sqrt(14)");
}

- (void)testNormalizingVectorX {
    // Normalizing vector (4, 0, 0) gives (1, 0, 0)
    Tuple v = { .components = {4.0, 0.0, 0.0, 0.0} };
    vector_float3 result = simd_normalize(v.components.xyz);
    XCTAssertTrue(simd_equal(result, (vector_float3){1.0, 0.0, 0.0}), @"Normalized (4,0,0) should be (1,0,0)");
}

- (void)testNormalizingVectorArbitrary {
    // Normalizing vector (1, 2, 3)
    Tuple v = { .components = {1.0, 2.0, 3.0, 0.0} };
    vector_float3 result = simd_normalize(v.components.xyz);
    XCTAssertEqualWithAccuracy(result.x, 0.26726, 0.0001, @"x should be ~0.26726");
    XCTAssertEqualWithAccuracy(result.y, 0.53452, 0.0001, @"y should be ~0.53452");
    XCTAssertEqualWithAccuracy(result.z, 0.80178, 0.0001, @"z should be ~0.80178");
}

- (void)testMagnitudeOfNormalizedVector {
    // The magnitude of a normalized vector is 1
    Tuple v = { .components = {1.0, 2.0, 3.0, 0.0} };
    vector_float3 norm = simd_normalize(v.components.xyz);
    float mag = simd_length(norm);
    XCTAssertEqualWithAccuracy(mag, 1.0, 0.0001, @"Magnitude of normalized vector should be 1");
}

- (void)testDotProduct {
    // The dot product of two tuples
    Tuple a = { .components = {1.0, 2.0, 3.0, 0.0} };
    Tuple b = { .components = {2.0, 3.0, 4.0, 0.0} };
    float result = simd_dot(a.components.xyz, b.components.xyz);
    XCTAssertEqual(result, 20.0, @"Dot product should be 20");
}

- (void)testCrossProduct {
    // The cross product of two vectors
    Tuple a = { .components = {1.0, 2.0, 3.0, 0.0} };
    Tuple b = { .components = {2.0, 3.0, 4.0, 0.0} };
    vector_float3 result = simd_cross(a.components.xyz, b.components.xyz);
    XCTAssertTrue(simd_equal(result, (vector_float3){-1.0, 2.0, -1.0}), @"Cross product a x b");
    
    result = simd_cross(b.components.xyz, a.components.xyz);
    XCTAssertTrue(simd_equal(result, (vector_float3){1.0, -2.0, 1.0}), @"Cross product b x a");
}

- (void)testReflectingVectorApproaching45Degrees {
    // Reflecting a vector approaching at 45 degrees
    Tuple v = { .components = {1.0, -1.0, 0.0, 0.0} };
    Tuple n = { .components = {0.0, 1.0, 0.0, 0.0} };
    // r = v - n * 2 * dot(v, n)
    float dot = simd_dot(v.components.xyz, n.components.xyz);
    vector_float3 result = v.components.xyz - n.components.xyz * 2.0 * dot;
    XCTAssertTrue(simd_all(result == (vector_float3){1.0, 1.0, 0.0}), @"Reflection at 45 degrees");
}

- (void)testReflectingVectorOffSlantedSurface {
    // Reflecting a vector off a slanted surface
    Tuple v = { .components = {0.0, -1.0, 0.0, 0.0} };
    Tuple n = { .components = {sqrtf(2.0)/2.0, sqrtf(2.0)/2.0, 0.0, 0.0} };
    float dot = simd_dot(v.components.xyz, n.components.xyz);
    vector_float3 result = v.components.xyz - n.components.xyz * 2.0 * dot;
    XCTAssertTrue(simd_all(simd_abs(result - (vector_float3){1.0, 0.0, 0.0}) < 0.0001), @"Reflection off slanted surface");
}

@end
