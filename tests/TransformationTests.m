#import <XCTest/XCTest.h>
#import "SharedTypes.h"

@interface TransformationTests : XCTestCase
@end

@implementation TransformationTests

// Helper to convert Matrix4x4 to matrix_float4x4 for multiplication
- (matrix_float4x4)toSIMD:(Matrix4x4)m {
    return matrix_from_columns(m.columns[0], m.columns[1], m.columns[2], m.columns[3]);
}

// Helper to multiply matrix by tuple
- (Tuple)multiplyMatrix:(Matrix4x4)m byTuple:(Tuple)t {
    matrix_float4x4 mat = [self toSIMD:m];
    Tuple result = { .components = matrix_multiply(mat, t.components) };
    return result;
}

// Helper to multiply two matrices
- (Matrix4x4)multiplyMatrix:(Matrix4x4)a byMatrix:(Matrix4x4)b {
    matrix_float4x4 ma = [self toSIMD:a];
    matrix_float4x4 mb = [self toSIMD:b];
    matrix_float4x4 mc = matrix_multiply(ma, mb);
    Matrix4x4 result;
    result.columns[0] = mc.columns[0];
    result.columns[1] = mc.columns[1];
    result.columns[2] = mc.columns[2];
    result.columns[3] = mc.columns[3];
    return result;
}

// Chapter 4 - Multiplying by a translation matrix
- (void)testTranslationMultiplyingPoint {
    Matrix4x4 transform = matrix_translation(5, -3, 2);
    Tuple p = { .components = {-3, 4, 5, 1} };
    Tuple result = [self multiplyMatrix:transform byTuple:p];
    
    XCTAssertEqualWithAccuracy(result.components.x, 2, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.y, 1, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.z, 7, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.w, 1, 0.0001);
}

// Chapter 4 - Multiplying by the inverse of a translation matrix
- (void)testTranslationMultiplyingByInverse {
    Matrix4x4 transform = matrix_translation(5, -3, 2);
    matrix_float4x4 mat = [self toSIMD:transform];
    matrix_float4x4 inv = matrix_invert(mat);
    Matrix4x4 inverse;
    inverse.columns[0] = inv.columns[0];
    inverse.columns[1] = inv.columns[1];
    inverse.columns[2] = inv.columns[2];
    inverse.columns[3] = inv.columns[3];
    
    Tuple p = { .components = {-3, 4, 5, 1} };
    Tuple result = [self multiplyMatrix:inverse byTuple:p];
    
    XCTAssertEqualWithAccuracy(result.components.x, -8, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.y, 7, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.z, 3, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.w, 1, 0.0001);
}

// Chapter 4 - Translation does not affect vectors
- (void)testTranslationDoesNotAffectVectors {
    Matrix4x4 transform = matrix_translation(5, -3, 2);
    Tuple v = { .components = {-3, 4, 5, 0} };
    Tuple result = [self multiplyMatrix:transform byTuple:v];
    
    XCTAssertEqualWithAccuracy(result.components.x, -3, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.y, 4, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.z, 5, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.w, 0, 0.0001);
}

// Chapter 4 - A scaling matrix applied to a point
- (void)testScalingAppliedToPoint {
    Matrix4x4 transform = matrix_scaling(2, 3, 4);
    Tuple p = { .components = {-4, 6, 8, 1} };
    Tuple result = [self multiplyMatrix:transform byTuple:p];
    
    XCTAssertEqualWithAccuracy(result.components.x, -8, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.y, 18, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.z, 32, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.w, 1, 0.0001);
}

// Chapter 4 - A scaling matrix applied to a vector
- (void)testScalingAppliedToVector {
    Matrix4x4 transform = matrix_scaling(2, 3, 4);
    Tuple v = { .components = {-4, 6, 8, 0} };
    Tuple result = [self multiplyMatrix:transform byTuple:v];
    
    XCTAssertEqualWithAccuracy(result.components.x, -8, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.y, 18, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.z, 32, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.w, 0, 0.0001);
}

// Chapter 4 - Multiplying by the inverse of a scaling matrix
- (void)testScalingMultiplyingByInverse {
    Matrix4x4 transform = matrix_scaling(2, 3, 4);
    matrix_float4x4 mat = [self toSIMD:transform];
    matrix_float4x4 inv = matrix_invert(mat);
    Matrix4x4 inverse;
    inverse.columns[0] = inv.columns[0];
    inverse.columns[1] = inv.columns[1];
    inverse.columns[2] = inv.columns[2];
    inverse.columns[3] = inv.columns[3];
    
    Tuple v = { .components = {-4, 6, 8, 0} };
    Tuple result = [self multiplyMatrix:inverse byTuple:v];
    
    XCTAssertEqualWithAccuracy(result.components.x, -2, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.y, 2, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.z, 2, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.w, 0, 0.0001);
}

// Chapter 4 - Reflection is scaling by a negative value
- (void)testReflectionIsScalingByNegative {
    Matrix4x4 transform = matrix_scaling(-1, 1, 1);
    Tuple p = { .components = {2, 3, 4, 1} };
    Tuple result = [self multiplyMatrix:transform byTuple:p];
    
    XCTAssertEqualWithAccuracy(result.components.x, -2, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.y, 3, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.z, 4, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.w, 1, 0.0001);
}

// Chapter 4 - Rotating a point around the X axis
- (void)testRotationAroundXAxis {
    Tuple p = { .components = {0, 1, 0, 1} };
    Matrix4x4 halfQuarter = matrix_rotation_x(M_PI_4);  // 45 degrees
    Matrix4x4 fullQuarter = matrix_rotation_x(M_PI_2);  // 90 degrees
    
    Tuple result1 = [self multiplyMatrix:halfQuarter byTuple:p];
    XCTAssertEqualWithAccuracy(result1.components.x, 0, 0.0001);
    XCTAssertEqualWithAccuracy(result1.components.y, sqrtf(2)/2, 0.0001);
    XCTAssertEqualWithAccuracy(result1.components.z, sqrtf(2)/2, 0.0001);
    
    Tuple result2 = [self multiplyMatrix:fullQuarter byTuple:p];
    XCTAssertEqualWithAccuracy(result2.components.x, 0, 0.0001);
    XCTAssertEqualWithAccuracy(result2.components.y, 0, 0.0001);
    XCTAssertEqualWithAccuracy(result2.components.z, 1, 0.0001);
}

// Chapter 4 - The inverse of an X rotation rotates in the opposite direction
- (void)testRotationXInverse {
    Tuple p = { .components = {0, 1, 0, 1} };
    Matrix4x4 halfQuarter = matrix_rotation_x(M_PI_4);
    matrix_float4x4 mat = [self toSIMD:halfQuarter];
    matrix_float4x4 inv = matrix_invert(mat);
    Matrix4x4 inverse;
    inverse.columns[0] = inv.columns[0];
    inverse.columns[1] = inv.columns[1];
    inverse.columns[2] = inv.columns[2];
    inverse.columns[3] = inv.columns[3];
    
    Tuple result = [self multiplyMatrix:inverse byTuple:p];
    XCTAssertEqualWithAccuracy(result.components.x, 0, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.y, sqrtf(2)/2, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.z, -sqrtf(2)/2, 0.0001);
}

// Chapter 4 - Rotating a point around the Y axis
- (void)testRotationAroundYAxis {
    Tuple p = { .components = {0, 0, 1, 1} };
    Matrix4x4 halfQuarter = matrix_rotation_y(M_PI_4);
    Matrix4x4 fullQuarter = matrix_rotation_y(M_PI_2);
    
    Tuple result1 = [self multiplyMatrix:halfQuarter byTuple:p];
    XCTAssertEqualWithAccuracy(result1.components.x, sqrtf(2)/2, 0.0001);
    XCTAssertEqualWithAccuracy(result1.components.y, 0, 0.0001);
    XCTAssertEqualWithAccuracy(result1.components.z, sqrtf(2)/2, 0.0001);
    
    Tuple result2 = [self multiplyMatrix:fullQuarter byTuple:p];
    XCTAssertEqualWithAccuracy(result2.components.x, 1, 0.0001);
    XCTAssertEqualWithAccuracy(result2.components.y, 0, 0.0001);
    XCTAssertEqualWithAccuracy(result2.components.z, 0, 0.0001);
}

// Chapter 4 - Rotating a point around the Z axis
- (void)testRotationAroundZAxis {
    Tuple p = { .components = {0, 1, 0, 1} };
    Matrix4x4 halfQuarter = matrix_rotation_z(M_PI_4);
    Matrix4x4 fullQuarter = matrix_rotation_z(M_PI_2);
    
    Tuple result1 = [self multiplyMatrix:halfQuarter byTuple:p];
    XCTAssertEqualWithAccuracy(result1.components.x, -sqrtf(2)/2, 0.0001);
    XCTAssertEqualWithAccuracy(result1.components.y, sqrtf(2)/2, 0.0001);
    XCTAssertEqualWithAccuracy(result1.components.z, 0, 0.0001);
    
    Tuple result2 = [self multiplyMatrix:fullQuarter byTuple:p];
    XCTAssertEqualWithAccuracy(result2.components.x, -1, 0.0001);
    XCTAssertEqualWithAccuracy(result2.components.y, 0, 0.0001);
    XCTAssertEqualWithAccuracy(result2.components.z, 0, 0.0001);
}

// Chapter 4 - A shearing transformation moves X in proportion to Y
- (void)testShearingXY {
    Matrix4x4 transform = matrix_shearing(1, 0, 0, 0, 0, 0);
    Tuple p = { .components = {2, 3, 4, 1} };
    Tuple result = [self multiplyMatrix:transform byTuple:p];
    
    XCTAssertEqualWithAccuracy(result.components.x, 5, 0.0001);  // 2 + 1*3 = 5
    XCTAssertEqualWithAccuracy(result.components.y, 3, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.z, 4, 0.0001);
}

// Chapter 4 - A shearing transformation moves X in proportion to Z
- (void)testShearingXZ {
    Matrix4x4 transform = matrix_shearing(0, 1, 0, 0, 0, 0);
    Tuple p = { .components = {2, 3, 4, 1} };
    Tuple result = [self multiplyMatrix:transform byTuple:p];
    
    XCTAssertEqualWithAccuracy(result.components.x, 6, 0.0001);  // 2 + 1*4 = 6
    XCTAssertEqualWithAccuracy(result.components.y, 3, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.z, 4, 0.0001);
}

// Chapter 4 - A shearing transformation moves Y in proportion to X
- (void)testShearingYX {
    Matrix4x4 transform = matrix_shearing(0, 0, 1, 0, 0, 0);
    Tuple p = { .components = {2, 3, 4, 1} };
    Tuple result = [self multiplyMatrix:transform byTuple:p];
    
    XCTAssertEqualWithAccuracy(result.components.x, 2, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.y, 5, 0.0001);  // 3 + 1*2 = 5
    XCTAssertEqualWithAccuracy(result.components.z, 4, 0.0001);
}

// Chapter 4 - A shearing transformation moves Y in proportion to Z
- (void)testShearingYZ {
    Matrix4x4 transform = matrix_shearing(0, 0, 0, 1, 0, 0);
    Tuple p = { .components = {2, 3, 4, 1} };
    Tuple result = [self multiplyMatrix:transform byTuple:p];
    
    XCTAssertEqualWithAccuracy(result.components.x, 2, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.y, 7, 0.0001);  // 3 + 1*4 = 7
    XCTAssertEqualWithAccuracy(result.components.z, 4, 0.0001);
}

// Chapter 4 - A shearing transformation moves Z in proportion to X
- (void)testShearingZX {
    Matrix4x4 transform = matrix_shearing(0, 0, 0, 0, 1, 0);
    Tuple p = { .components = {2, 3, 4, 1} };
    Tuple result = [self multiplyMatrix:transform byTuple:p];
    
    XCTAssertEqualWithAccuracy(result.components.x, 2, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.y, 3, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.z, 6, 0.0001);  // 4 + 1*2 = 6
}

// Chapter 4 - A shearing transformation moves Z in proportion to Y
- (void)testShearingZY {
    Matrix4x4 transform = matrix_shearing(0, 0, 0, 0, 0, 1);
    Tuple p = { .components = {2, 3, 4, 1} };
    Tuple result = [self multiplyMatrix:transform byTuple:p];
    
    XCTAssertEqualWithAccuracy(result.components.x, 2, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.y, 3, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.z, 7, 0.0001);  // 4 + 1*3 = 7
}

// Chapter 4 - Individual transformations are applied in sequence
- (void)testIndividualTransformationsSequence {
    Tuple p = { .components = {1, 0, 1, 1} };
    Matrix4x4 A = matrix_rotation_x(M_PI_2);
    Matrix4x4 B = matrix_scaling(5, 5, 5);
    Matrix4x4 C = matrix_translation(10, 5, 7);
    
    // Apply rotation first
    Tuple p2 = [self multiplyMatrix:A byTuple:p];
    XCTAssertEqualWithAccuracy(p2.components.x, 1, 0.0001);
    XCTAssertEqualWithAccuracy(p2.components.y, -1, 0.0001);
    XCTAssertEqualWithAccuracy(p2.components.z, 0, 0.0001);
    
    // Then scaling
    Tuple p3 = [self multiplyMatrix:B byTuple:p2];
    XCTAssertEqualWithAccuracy(p3.components.x, 5, 0.0001);
    XCTAssertEqualWithAccuracy(p3.components.y, -5, 0.0001);
    XCTAssertEqualWithAccuracy(p3.components.z, 0, 0.0001);
    
    // Then translation
    Tuple p4 = [self multiplyMatrix:C byTuple:p3];
    XCTAssertEqualWithAccuracy(p4.components.x, 15, 0.0001);
    XCTAssertEqualWithAccuracy(p4.components.y, 0, 0.0001);
    XCTAssertEqualWithAccuracy(p4.components.z, 7, 0.0001);
}

// Chapter 4 - Chained transformations must be applied in reverse order
- (void)testChainedTransformations {
    Tuple p = { .components = {1, 0, 1, 1} };
    Matrix4x4 A = matrix_rotation_x(M_PI_2);
    Matrix4x4 B = matrix_scaling(5, 5, 5);
    Matrix4x4 C = matrix_translation(10, 5, 7);
    
    // C * B * A means apply A, then B, then C (reverse order)
    Matrix4x4 T = [self multiplyMatrix:C byMatrix:[self multiplyMatrix:B byMatrix:A]];
    Tuple result = [self multiplyMatrix:T byTuple:p];
    
    XCTAssertEqualWithAccuracy(result.components.x, 15, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.y, 0, 0.0001);
    XCTAssertEqualWithAccuracy(result.components.z, 7, 0.0001);
}

@end
