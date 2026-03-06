#import <XCTest/XCTest.h>
#import "SharedTypes.h"

@interface MatrixTests : XCTestCase
@end

@implementation MatrixTests

// Helper to create matrix from row-major array
- (Matrix4x4)matrixFromArray:(const float[16])values {
    Matrix4x4 m;
    // Metal/SIMD uses column-major order
    // values[row*4 + col] -> m.columns[col][row]
    m.columns[0] = (vector_float4){values[0], values[4], values[8], values[12]};
    m.columns[1] = (vector_float4){values[1], values[5], values[9], values[13]};
    m.columns[2] = (vector_float4){values[2], values[6], values[10], values[14]};
    m.columns[3] = (vector_float4){values[3], values[7], values[11], values[15]};
    return m;
}

// Helper to get matrix element (row, col) from column-major matrix
- (float)getMatrixElementRow:(int)row col:(int)col fromMatrix:(Matrix4x4)m {
    return m.columns[col][row];
}

// Helper to check matrix equality with tolerance
- (BOOL)matrix:(Matrix4x4)a equalsMatrix:(Matrix4x4)b tolerance:(float)tol {
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            if (fabs(a.columns[col][row] - b.columns[col][row]) > tol) {
                return NO;
            }
        }
    }
    return YES;
}

// Chapter 3 - Constructing and inspecting a 4x4 matrix
- (void)testConstructing4x4Matrix {
    const float values[16] = {
        1, 2, 3, 4,
        5.5, 6.5, 7.5, 8.5,
        9, 10, 11, 12,
        13.5, 14.5, 15.5, 16.5
    };
    Matrix4x4 m = [self matrixFromArray:values];
    
    XCTAssertEqualWithAccuracy([self getMatrixElementRow:0 col:0 fromMatrix:m], 1, 0.0001);
    XCTAssertEqualWithAccuracy([self getMatrixElementRow:0 col:3 fromMatrix:m], 4, 0.0001);
    XCTAssertEqualWithAccuracy([self getMatrixElementRow:1 col:0 fromMatrix:m], 5.5, 0.0001);
    XCTAssertEqualWithAccuracy([self getMatrixElementRow:1 col:2 fromMatrix:m], 7.5, 0.0001);
    XCTAssertEqualWithAccuracy([self getMatrixElementRow:2 col:2 fromMatrix:m], 11, 0.0001);
    XCTAssertEqualWithAccuracy([self getMatrixElementRow:3 col:0 fromMatrix:m], 13.5, 0.0001);
    XCTAssertEqualWithAccuracy([self getMatrixElementRow:3 col:2 fromMatrix:m], 15.5, 0.0001);
}

// Chapter 3 - A 4x4 matrix is equal to itself
- (void)test4x4MatrixEquality {
    const float values[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 8, 7, 6,
        5, 4, 3, 2
    };
    Matrix4x4 a = [self matrixFromArray:values];
    Matrix4x4 b = [self matrixFromArray:values];
    
    XCTAssertTrue([self matrix:a equalsMatrix:b tolerance:0.0001], @"Matrices should be equal");
}

// Chapter 3 - A 4x4 matrix is not equal to a different matrix
- (void)test4x4MatrixInequality {
    const float a_values[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 8, 7, 6,
        5, 4, 3, 2
    };
    const float b_values[16] = {
        2, 3, 4, 5,
        6, 7, 8, 9,
        8, 7, 6, 5,
        4, 3, 2, 1
    };
    Matrix4x4 a = [self matrixFromArray:a_values];
    Matrix4x4 b = [self matrixFromArray:b_values];
    
    XCTAssertFalse([self matrix:a equalsMatrix:b tolerance:0.0001], @"Matrices should be different");
}

// Chapter 3 - Multiplying two 4x4 matrices
- (void)testMultiplyingTwoMatrices {
    const float a_values[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 8, 7, 6,
        5, 4, 3, 2
    };
    const float b_values[16] = {
        -2, 1, 2, 3,
        3, 2, 1, -1,
        4, 3, 6, 5,
        1, 2, 7, 8
    };
    Matrix4x4 a = [self matrixFromArray:a_values];
    Matrix4x4 b = [self matrixFromArray:b_values];
    
    // Convert to SIMD matrices and multiply
    matrix_float4x4 ma = matrix_from_columns(a.columns[0], a.columns[1], a.columns[2], a.columns[3]);
    matrix_float4x4 mb = matrix_from_columns(b.columns[0], b.columns[1], b.columns[2], b.columns[3]);
    matrix_float4x4 mc = matrix_multiply(ma, mb);
    
    // Expected result (row-major from book)
    // Column 0: rows 0-3 -> expected[0], expected[4], expected[8], expected[12]
    // Column 1: rows 0-3 -> expected[1], expected[5], expected[9], expected[13]
    // etc.
    const float expected[16] = {
        20, 22, 50, 48,
        44, 54, 114, 108,
        40, 58, 110, 102,
        16, 26, 46, 42
    };
    
    // mc.columns[col][row] corresponds to expected[row*4 + col]
    XCTAssertEqualWithAccuracy(mc.columns[0][0], expected[0*4 + 0], 0.0001);   // row 0, col 0
    XCTAssertEqualWithAccuracy(mc.columns[0][1], expected[1*4 + 0], 0.0001);   // row 1, col 0
    XCTAssertEqualWithAccuracy(mc.columns[0][2], expected[2*4 + 0], 0.0001);   // row 2, col 0
    XCTAssertEqualWithAccuracy(mc.columns[0][3], expected[3*4 + 0], 0.0001);   // row 3, col 0
    XCTAssertEqualWithAccuracy(mc.columns[1][0], expected[0*4 + 1], 0.0001);   // row 0, col 1
    XCTAssertEqualWithAccuracy(mc.columns[1][1], expected[1*4 + 1], 0.0001);   // row 1, col 1
    XCTAssertEqualWithAccuracy(mc.columns[1][2], expected[2*4 + 1], 0.0001);   // row 2, col 1
    XCTAssertEqualWithAccuracy(mc.columns[1][3], expected[3*4 + 1], 0.0001);   // row 3, col 1
    XCTAssertEqualWithAccuracy(mc.columns[2][0], expected[0*4 + 2], 0.0001);   // row 0, col 2
    XCTAssertEqualWithAccuracy(mc.columns[2][1], expected[1*4 + 2], 0.0001);   // row 1, col 2
    XCTAssertEqualWithAccuracy(mc.columns[2][2], expected[2*4 + 2], 0.0001);   // row 2, col 2
    XCTAssertEqualWithAccuracy(mc.columns[2][3], expected[3*4 + 2], 0.0001);   // row 3, col 2
    XCTAssertEqualWithAccuracy(mc.columns[3][0], expected[0*4 + 3], 0.0001);   // row 0, col 3
    XCTAssertEqualWithAccuracy(mc.columns[3][1], expected[1*4 + 3], 0.0001);   // row 1, col 3
    XCTAssertEqualWithAccuracy(mc.columns[3][2], expected[2*4 + 3], 0.0001);   // row 2, col 3
    XCTAssertEqualWithAccuracy(mc.columns[3][3], expected[3*4 + 3], 0.0001);   // row 3, col 3
}

// Chapter 3 - A matrix multiplied by a tuple
- (void)testMatrixMultipliedByTuple {
    const float m_values[16] = {
        1, 2, 3, 4,
        2, 4, 4, 2,
        8, 6, 4, 1,
        0, 0, 0, 1
    };
    Matrix4x4 m = [self matrixFromArray:m_values];
    Tuple t = { .components = {1, 2, 3, 1} };
    
    matrix_float4x4 mat = matrix_from_columns(m.columns[0], m.columns[1], m.columns[2], m.columns[3]);
    vector_float4 result = matrix_multiply(mat, t.components);
    
    XCTAssertEqualWithAccuracy(result.x, 18, 0.0001);
    XCTAssertEqualWithAccuracy(result.y, 24, 0.0001);
    XCTAssertEqualWithAccuracy(result.z, 33, 0.0001);
    XCTAssertEqualWithAccuracy(result.w, 1, 0.0001);
}

// Chapter 3 - Multiplying a matrix by the identity matrix
- (void)testMatrixMultipliedByIdentity {
    const float a_values[16] = {
        0, 1, 2, 4,
        1, 2, 4, 8,
        2, 4, 8, 16,
        4, 8, 16, 32
    };
    Matrix4x4 a = [self matrixFromArray:a_values];
    Matrix4x4 identity = MATRIX4X4_IDENTITY;
    
    matrix_float4x4 ma = matrix_from_columns(a.columns[0], a.columns[1], a.columns[2], a.columns[3]);
    matrix_float4x4 mi = matrix_from_columns(identity.columns[0], identity.columns[1], 
                                              identity.columns[2], identity.columns[3]);
    matrix_float4x4 result = matrix_multiply(ma, mi);
    
    // Result should equal original matrix
    XCTAssertTrue([self matrix:a equalsMatrix:*((Matrix4x4 *)&result) tolerance:0.0001], 
                  @"A * I should equal A");
}

// Chapter 3 - Transposing a matrix
- (void)testTransposingMatrix {
    const float a_values[16] = {
        0, 9, 3, 0,
        9, 8, 0, 8,
        1, 8, 5, 3,
        0, 0, 5, 8
    };
    Matrix4x4 a = [self matrixFromArray:a_values];
    
    matrix_float4x4 ma = matrix_from_columns(a.columns[0], a.columns[1], a.columns[2], a.columns[3]);
    matrix_float4x4 transposed = matrix_transpose(ma);
    
    // Expected transposed result (row-major from book)
    const float expected[16] = {
        0, 9, 1, 0,
        9, 8, 8, 0,
        3, 0, 5, 5,
        0, 8, 3, 8
    };
    
    // transposed.columns[col][row] corresponds to expected[row*4 + col]
    XCTAssertEqualWithAccuracy(transposed.columns[0][0], expected[0*4 + 0], 0.0001);
    XCTAssertEqualWithAccuracy(transposed.columns[0][1], expected[1*4 + 0], 0.0001);
    XCTAssertEqualWithAccuracy(transposed.columns[0][2], expected[2*4 + 0], 0.0001);
    XCTAssertEqualWithAccuracy(transposed.columns[0][3], expected[3*4 + 0], 0.0001);
    XCTAssertEqualWithAccuracy(transposed.columns[1][0], expected[0*4 + 1], 0.0001);
    XCTAssertEqualWithAccuracy(transposed.columns[1][1], expected[1*4 + 1], 0.0001);
    XCTAssertEqualWithAccuracy(transposed.columns[1][2], expected[2*4 + 1], 0.0001);
    XCTAssertEqualWithAccuracy(transposed.columns[1][3], expected[3*4 + 1], 0.0001);
    XCTAssertEqualWithAccuracy(transposed.columns[2][0], expected[0*4 + 2], 0.0001);
    XCTAssertEqualWithAccuracy(transposed.columns[2][1], expected[1*4 + 2], 0.0001);
    XCTAssertEqualWithAccuracy(transposed.columns[2][2], expected[2*4 + 2], 0.0001);
    XCTAssertEqualWithAccuracy(transposed.columns[2][3], expected[3*4 + 2], 0.0001);
    XCTAssertEqualWithAccuracy(transposed.columns[3][0], expected[0*4 + 3], 0.0001);
    XCTAssertEqualWithAccuracy(transposed.columns[3][1], expected[1*4 + 3], 0.0001);
    XCTAssertEqualWithAccuracy(transposed.columns[3][2], expected[2*4 + 3], 0.0001);
    XCTAssertEqualWithAccuracy(transposed.columns[3][3], expected[3*4 + 3], 0.0001);
}

// Chapter 3 - Transposing the identity matrix
- (void)testTransposingIdentity {
    Matrix4x4 identity = MATRIX4X4_IDENTITY;
    matrix_float4x4 mi = matrix_from_columns(identity.columns[0], identity.columns[1], 
                                              identity.columns[2], identity.columns[3]);
    matrix_float4x4 transposed = matrix_transpose(mi);
    
    // Transposed identity should equal identity
    XCTAssertTrue([self matrix:identity equalsMatrix:*((Matrix4x4 *)&transposed) tolerance:0.0001],
                  @"Transpose of I should be I");
}

// Chapter 3 - Calculating the inverse of a matrix
- (void)testCalculatingInverse {
    const float a_values[16] = {
        -5, 2, 6, -8,
        1, -5, 1, 8,
        7, 7, -6, -7,
        1, -3, 7, 4
    };
    Matrix4x4 a = [self matrixFromArray:a_values];
    
    matrix_float4x4 ma = matrix_from_columns(a.columns[0], a.columns[1], a.columns[2], a.columns[3]);
    matrix_float4x4 inverse = matrix_invert(ma);
    
    // Multiply a * inverse should give identity
    matrix_float4x4 product = matrix_multiply(ma, inverse);
    matrix_float4x4 identity = matrix_identity_float4x4;
    
    for (int i = 0; i < 4; i++) {
        XCTAssertTrue(simd_all(simd_abs(product.columns[i] - identity.columns[i]) < 0.001), 
                      @"A * A^-1 should equal I");
    }
}

// Chapter 3 - Multiplying a product by its inverse
- (void)testMultiplyProductByInverse {
    const float a_values[16] = {
        3, -9, 7, 3,
        3, -8, 2, -9,
        -4, 4, 4, 1,
        -6, 5, -1, 1
    };
    const float b_values[16] = {
        8, 2, 2, 2,
        3, -1, 7, 0,
        7, 0, 5, 4,
        6, -2, 0, 5
    };
    Matrix4x4 a = [self matrixFromArray:a_values];
    Matrix4x4 b = [self matrixFromArray:b_values];
    
    matrix_float4x4 ma = matrix_from_columns(a.columns[0], a.columns[1], a.columns[2], a.columns[3]);
    matrix_float4x4 mb = matrix_from_columns(b.columns[0], b.columns[1], b.columns[2], b.columns[3]);
    matrix_float4x4 c = matrix_multiply(ma, mb);
    matrix_float4x4 b_inverse = matrix_invert(mb);
    matrix_float4x4 result = matrix_multiply(c, b_inverse);
    
    // result should equal a
    for (int i = 0; i < 4; i++) {
        XCTAssertTrue(simd_all(simd_abs(result.columns[i] - ma.columns[i]) < 0.001), 
                      @"C * B^-1 should equal A");
    }
}

@end
