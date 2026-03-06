#import <XCTest/XCTest.h>
#import "SharedTypes.h"

@interface TupleTests : XCTestCase
@end

@implementation TupleTests

// Chapter 1 - Tuples, Points, and Vectors

- (void)testTupleIsPoint {
    // A tuple with w=1.0 is a point
    Tuple a = { .components = {4.3, -4.2, 3.1, 1.0} };
    XCTAssertEqualWithAccuracy(a.components.x, 4.3, 0.0001, @"x should be 4.3");
    XCTAssertEqualWithAccuracy(a.components.y, -4.2, 0.0001, @"y should be -4.2");
    XCTAssertEqualWithAccuracy(a.components.z, 3.1, 0.0001, @"z should be 3.1");
    XCTAssertEqual(a.components.w, 1.0, @"w should be 1.0 (point)");
}

- (void)testTupleIsVector {
    // A tuple with w=0.0 is a vector
    Tuple a = { .components = {4.3, -4.2, 3.1, 0.0} };
    XCTAssertEqualWithAccuracy(a.components.x, 4.3, 0.0001, @"x should be 4.3");
    XCTAssertEqualWithAccuracy(a.components.y, -4.2, 0.0001, @"y should be -4.2");
    XCTAssertEqualWithAccuracy(a.components.z, 3.1, 0.0001, @"z should be 3.1");
    XCTAssertEqual(a.components.w, 0.0, @"w should be 0.0 (vector)");
}

- (void)testPointCreatesTupleWithW1 {
    // Point() creates tuples with w=1
    Tuple p = { .components = {4.0, -4.0, 3.0, TUPLE_POINT} };
    XCTAssertTrue(simd_equal(p.components, (vector_float4){4.0, -4.0, 3.0, 1.0}), @"Should be a point");
}

- (void)testVectorCreatesTupleWithW0 {
    // Vector() creates tuples with w=0
    Tuple v = { .components = {4.0, -4.0, 3.0, TUPLE_VECTOR} };
    XCTAssertTrue(simd_equal(v.components, (vector_float4){4.0, -4.0, 3.0, 0.0}), @"Should be a vector");
}

- (void)testAddingTwoTuples {
    // Adding two tuples
    Tuple a1 = { .components = {3.0, -2.0, 5.0, 1.0} };
    Tuple a2 = { .components = {-2.0, 3.0, 1.0, 0.0} };
    vector_float4 result = a1.components + a2.components;
    XCTAssertTrue(simd_equal(result, (vector_float4){1.0, 1.0, 6.0, 1.0}), @"Adding point and vector");
}

- (void)testSubtractingTwoPoints {
    // Subtracting two points
    Tuple p1 = { .components = {3.0, 2.0, 1.0, 1.0} };
    Tuple p2 = { .components = {5.0, 6.0, 7.0, 1.0} };
    vector_float4 result = p1.components - p2.components;
    XCTAssertTrue(simd_equal(result, (vector_float4){-2.0, -4.0, -6.0, 0.0}), @"Subtracting points gives vector");
}

- (void)testSubtractingVectorFromPoint {
    // Subtracting a vector from a point
    Tuple p = { .components = {3.0, 2.0, 1.0, 1.0} };
    Tuple v = { .components = {5.0, 6.0, 7.0, 0.0} };
    vector_float4 result = p.components - v.components;
    XCTAssertTrue(simd_equal(result, (vector_float4){-2.0, -4.0, -6.0, 1.0}), @"Point minus vector is point");
}

- (void)testSubtractingTwoVectors {
    // Subtracting two vectors
    Tuple v1 = { .components = {3.0, 2.0, 1.0, 0.0} };
    Tuple v2 = { .components = {5.0, 6.0, 7.0, 0.0} };
    vector_float4 result = v1.components - v2.components;
    XCTAssertTrue(simd_equal(result, (vector_float4){-2.0, -4.0, -6.0, 0.0}), @"Vector minus vector is vector");
}

- (void)testNegatingTuple {
    // Negating a tuple
    Tuple a = { .components = {1.0, -2.0, 3.0, -4.0} };
    vector_float4 result = -a.components;
    XCTAssertTrue(simd_equal(result, (vector_float4){-1.0, 2.0, -3.0, 4.0}), @"Negation");
}

- (void)testMultiplyingTupleByScalar {
    // Multiplying a tuple by a scalar
    Tuple a = { .components = {1.0, -2.0, 3.0, -4.0} };
    vector_float4 result = a.components * 3.5;
    XCTAssertTrue(simd_equal(result, (vector_float4){3.5, -7.0, 10.5, -14.0}), @"Scalar multiplication");
}

- (void)testMultiplyingTupleByFraction {
    // Multiplying a tuple by a fraction
    Tuple a = { .components = {1.0, -2.0, 3.0, -4.0} };
    vector_float4 result = a.components * 0.5;
    XCTAssertTrue(simd_equal(result, (vector_float4){0.5, -1.0, 1.5, -2.0}), @"Fraction multiplication");
}

- (void)testDividingTupleByScalar {
    // Dividing a tuple by a scalar
    Tuple a = { .components = {1.0, -2.0, 3.0, -4.0} };
    vector_float4 result = a.components / 2.0;
    XCTAssertTrue(simd_equal(result, (vector_float4){0.5, -1.0, 1.5, -2.0}), @"Scalar division");
}

@end
