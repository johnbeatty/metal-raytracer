#import <XCTest/XCTest.h>
#import "SharedTypes.h"

@interface PatternTests : XCTestCase
@end

@implementation PatternTests

// Helper to check if two vectors are approximately equal
- (BOOL)vector:(vector_float4)a equalsVector:(vector_float4)b tolerance:(float)tol {
    return simd_all(simd_abs(a - b) < tol);
}

// MARK: - Stripe Pattern Tests

// Chapter 10 - Creating a stripe pattern
- (void)testCreatingStripePattern {
    vector_float4 white = {1, 1, 1, 1};
    vector_float4 black = {0, 0, 0, 1};
    SurfacePattern pattern = pattern_stripe(white, black);
    
    XCTAssertTrue([self vector:pattern.a equalsVector:white tolerance:0.0001]);
    XCTAssertTrue([self vector:pattern.b equalsVector:black tolerance:0.0001]);
}

// Chapter 10 - A stripe pattern is constant in Y
- (void)testStripePatternIsConstantInY {
    vector_float4 white = {1, 1, 1, 1};
    vector_float4 black = {0, 0, 0, 1};
    SurfacePattern pattern = pattern_stripe(white, black);
    
    vector_float4 p1 = {0, 0, 0, 1};
    vector_float4 p2 = {0, 1, 0, 1};
    vector_float4 p3 = {0, 2, 0, 1};
    
    XCTAssertTrue([self vector:pattern_stripe_at(pattern, p1) equalsVector:white tolerance:0.0001]);
    XCTAssertTrue([self vector:pattern_stripe_at(pattern, p2) equalsVector:white tolerance:0.0001]);
    XCTAssertTrue([self vector:pattern_stripe_at(pattern, p3) equalsVector:white tolerance:0.0001]);
}

// Chapter 10 - A stripe pattern is constant in Z
- (void)testStripePatternIsConstantInZ {
    vector_float4 white = {1, 1, 1, 1};
    vector_float4 black = {0, 0, 0, 1};
    SurfacePattern pattern = pattern_stripe(white, black);
    
    vector_float4 p1 = {0, 0, 0, 1};
    vector_float4 p2 = {0, 0, 1, 1};
    vector_float4 p3 = {0, 0, 2, 1};
    
    XCTAssertTrue([self vector:pattern_stripe_at(pattern, p1) equalsVector:white tolerance:0.0001]);
    XCTAssertTrue([self vector:pattern_stripe_at(pattern, p2) equalsVector:white tolerance:0.0001]);
    XCTAssertTrue([self vector:pattern_stripe_at(pattern, p3) equalsVector:white tolerance:0.0001]);
}

// Chapter 10 - A stripe pattern alternates in X
- (void)testStripePatternAlternatesInX {
    vector_float4 white = {1, 1, 1, 1};
    vector_float4 black = {0, 0, 0, 1};
    SurfacePattern pattern = pattern_stripe(white, black);
    
    vector_float4 p1 = {0, 0, 0, 1};   // Even - white
    vector_float4 p2 = {0.9, 0, 0, 1}; // Still in even stripe
    vector_float4 p3 = {1, 0, 0, 1};   // Odd - black
    vector_float4 p4 = {-0.1, 0, 0, 1}; // Negative, but still white
    vector_float4 p5 = {-1, 0, 0, 1};  // Negative odd - black
    
    XCTAssertTrue([self vector:pattern_stripe_at(pattern, p1) equalsVector:white tolerance:0.0001]);
    XCTAssertTrue([self vector:pattern_stripe_at(pattern, p2) equalsVector:white tolerance:0.0001]);
    XCTAssertTrue([self vector:pattern_stripe_at(pattern, p3) equalsVector:black tolerance:0.0001]);
    XCTAssertTrue([self vector:pattern_stripe_at(pattern, p4) equalsVector:white tolerance:0.0001]);
    XCTAssertTrue([self vector:pattern_stripe_at(pattern, p5) equalsVector:black tolerance:0.0001]);
}

// MARK: - Gradient Pattern Tests

// Chapter 10 - A gradient linearly interpolates between colors
- (void)testGradientPatternLinearlyInterpolates {
    vector_float4 white = {1, 1, 1, 1};
    vector_float4 black = {0, 0, 0, 1};
    SurfacePattern pattern = pattern_gradient(white, black);
    
    vector_float4 p1 = {0, 0, 0, 1};  // Should be white
    vector_float4 p2 = {0.25, 0, 0, 1}; // Should be 75% white, 25% black
    vector_float4 p3 = {0.5, 0, 0, 1};  // Should be 50% white, 50% black
    vector_float4 p4 = {0.75, 0, 0, 1}; // Should be 25% white, 75% black
    
    vector_float4 result1 = pattern_gradient_at(pattern, p1);
    vector_float4 result2 = pattern_gradient_at(pattern, p2);
    vector_float4 result3 = pattern_gradient_at(pattern, p3);
    vector_float4 result4 = pattern_gradient_at(pattern, p4);
    
    vector_float4 expected2 = {0.75, 0.75, 0.75, 1};
    vector_float4 expected3 = {0.5, 0.5, 0.5, 1};
    vector_float4 expected4 = {0.25, 0.25, 0.25, 1};
    
    XCTAssertTrue([self vector:result1 equalsVector:white tolerance:0.0001]);
    XCTAssertTrue([self vector:result2 equalsVector:expected2 tolerance:0.0001]);
    XCTAssertTrue([self vector:result3 equalsVector:expected3 tolerance:0.0001]);
    XCTAssertTrue([self vector:result4 equalsVector:expected4 tolerance:0.0001]);
}

// MARK: - Ring Pattern Tests

// Chapter 10 - A ring should extend in both X and Z
- (void)testRingPatternExtendsInXAndZ {
    vector_float4 white = {1, 1, 1, 1};
    vector_float4 black = {0, 0, 0, 1};
    SurfacePattern pattern = pattern_ring(white, black);
    
    vector_float4 p1 = {0, 0, 0, 1};    // Center - white
    vector_float4 p2 = {1, 0, 0, 1};    // Distance 1 - black
    vector_float4 p3 = {0, 0, 1, 1};    // Distance 1 - black
    vector_float4 p4 = {0.708, 0, 0.708, 1}; // Distance ~1 - black (sqrt(0.5^2 + 0.5^2) is wrong, sqrt(0.708^2 + 0.708^2) ≈ 1)
    
    // p1 is at distance 0, should be white
    XCTAssertTrue([self vector:pattern_ring_at(pattern, p1) equalsVector:white tolerance:0.0001]);
    // p2 is at distance 1, should be black
    XCTAssertTrue([self vector:pattern_ring_at(pattern, p2) equalsVector:black tolerance:0.0001]);
    // p3 is at distance 1, should be black
    XCTAssertTrue([self vector:pattern_ring_at(pattern, p3) equalsVector:black tolerance:0.0001]);
    // p4 is at distance sqrt(0.708^2 + 0.708^2) ≈ 1.0, should be black
    XCTAssertTrue([self vector:pattern_ring_at(pattern, p4) equalsVector:black tolerance:0.0001]);
}

// MARK: - Checker Pattern Tests

// Chapter 10 - Checkers should repeat in X
- (void)testCheckerPatternRepeatsInX {
    vector_float4 white = {1, 1, 1, 1};
    vector_float4 black = {0, 0, 0, 1};
    SurfacePattern pattern = pattern_checker(white, black);
    
    vector_float4 p1 = {0, 0, 0, 1};  // Even - white
    vector_float4 p2 = {0.99, 0, 0, 1}; // Still in first unit - white
    vector_float4 p3 = {1.01, 0, 0, 1}; // In second unit - black
    
    XCTAssertTrue([self vector:pattern_checker_at(pattern, p1) equalsVector:white tolerance:0.0001]);
    XCTAssertTrue([self vector:pattern_checker_at(pattern, p2) equalsVector:white tolerance:0.0001]);
    XCTAssertTrue([self vector:pattern_checker_at(pattern, p3) equalsVector:black tolerance:0.0001]);
}

// Chapter 10 - Checkers should repeat in Y
- (void)testCheckerPatternRepeatsInY {
    vector_float4 white = {1, 1, 1, 1};
    vector_float4 black = {0, 0, 0, 1};
    SurfacePattern pattern = pattern_checker(white, black);
    
    vector_float4 p1 = {0, 0, 0, 1};  // Even - white
    vector_float4 p2 = {0, 0.99, 0, 1}; // Still in first unit - white
    vector_float4 p3 = {0, 1.01, 0, 1}; // In second unit - black
    
    XCTAssertTrue([self vector:pattern_checker_at(pattern, p1) equalsVector:white tolerance:0.0001]);
    XCTAssertTrue([self vector:pattern_checker_at(pattern, p2) equalsVector:white tolerance:0.0001]);
    XCTAssertTrue([self vector:pattern_checker_at(pattern, p3) equalsVector:black tolerance:0.0001]);
}

// Chapter 10 - Checkers should repeat in Z
- (void)testCheckerPatternRepeatsInZ {
    vector_float4 white = {1, 1, 1, 1};
    vector_float4 black = {0, 0, 0, 1};
    SurfacePattern pattern = pattern_checker(white, black);
    
    vector_float4 p1 = {0, 0, 0, 1};  // Even - white
    vector_float4 p2 = {0, 0, 0.99, 1}; // Still in first unit - white
    vector_float4 p3 = {0, 0, 1.01, 1}; // In second unit - black
    
    XCTAssertTrue([self vector:pattern_checker_at(pattern, p1) equalsVector:white tolerance:0.0001]);
    XCTAssertTrue([self vector:pattern_checker_at(pattern, p2) equalsVector:white tolerance:0.0001]);
    XCTAssertTrue([self vector:pattern_checker_at(pattern, p3) equalsVector:black tolerance:0.0001]);
}

// MARK: - Pattern Transformation Tests

// Chapter 10 - Stripes with an object transformation
- (void)testStripePatternWithObjectTransformation {
    // Create a sphere with scaling transform
    Sphere s = sphere_create(1);
    Matrix4x4 scale = matrix_scaling(2, 2, 2);
    sphere_set_transform(&s, scale);
    
    // Create stripe pattern
    vector_float4 white = {1, 1, 1, 1};
    vector_float4 black = {0, 0, 0, 1};
    SurfacePattern pattern = pattern_stripe(white, black);
    
    // Point at (1.5, 0, 0) in world space
    // After inverse sphere transform (scale by 0.5), becomes (0.75, 0, 0) in object space
    // 0.75 is still in white stripe (floor(0.75) = 0, which is even)
    vector_float4 world_point = {1.5, 0, 0, 1};
    vector_float4 result = pattern_at_object(pattern, s.inverseTransform, world_point);
    
    XCTAssertTrue([self vector:result equalsVector:white tolerance:0.0001]);
}

// Chapter 10 - Stripes with a pattern transformation
- (void)testStripePatternWithPatternTransformation {
    // Create stripe pattern with scaling transform
    vector_float4 white = {1, 1, 1, 1};
    vector_float4 black = {0, 0, 0, 1};
    SurfacePattern pattern = pattern_stripe(white, black);
    Matrix4x4 scale = matrix_scaling(2, 2, 2);
    pattern_set_transform(&pattern, scale);
    
    // Point at (2, 0, 0) in pattern space
    // After inverse pattern transform (scale by 0.5), becomes (1, 0, 0)
    // 1 is in black stripe (floor(1) = 1, which is odd)
    vector_float4 point = {2, 0, 0, 1};
    vector_float4 result = pattern_at(pattern, point);
    
    XCTAssertTrue([self vector:result equalsVector:black tolerance:0.0001]);
}

// Chapter 10 - Stripes with both object and pattern transformation
- (void)testStripePatternWithObjectAndPatternTransformation {
    // Create sphere with translation
    Sphere s = sphere_create(1);
    Matrix4x4 trans = matrix_translation(0.5, 0, 0);
    sphere_set_transform(&s, trans);
    
    // Create stripe pattern with scaling
    vector_float4 white = {1, 1, 1, 1};
    vector_float4 black = {0, 0, 0, 1};
    SurfacePattern pattern = pattern_stripe(white, black);
    Matrix4x4 scale = matrix_scaling(2, 2, 2);
    pattern_set_transform(&pattern, scale);
    
    // Point at (2.5, 0, 0) in world space
    // After inverse sphere transform (translate by -0.5), becomes (2, 0, 0) in object space
    // After inverse pattern transform (scale by 0.5), becomes (1, 0, 0) in pattern space
    // 1 is in black stripe
    vector_float4 world_point = {2.5, 0, 0, 1};
    vector_float4 result = pattern_at_object(pattern, s.inverseTransform, world_point);
    
    XCTAssertTrue([self vector:result equalsVector:black tolerance:0.0001]);
}

// MARK: - Pattern at Point Tests

// Chapter 10 - Testing pattern_at function with stripe pattern
- (void)testPatternAtWithStripePattern {
    vector_float4 white = {1, 1, 1, 1};
    vector_float4 black = {0, 0, 0, 1};
    SurfacePattern pattern = pattern_stripe(white, black);
    
    vector_float4 p1 = {0, 0, 0, 1};
    vector_float4 p2 = {1, 0, 0, 1};
    
    XCTAssertTrue([self vector:pattern_at(pattern, p1) equalsVector:white tolerance:0.0001]);
    XCTAssertTrue([self vector:pattern_at(pattern, p2) equalsVector:black tolerance:0.0001]);
}

// Chapter 10 - Testing pattern_at function with gradient pattern
- (void)testPatternAtWithGradientPattern {
    vector_float4 white = {1, 1, 1, 1};
    vector_float4 black = {0, 0, 0, 1};
    SurfacePattern pattern = pattern_gradient(white, black);
    
    vector_float4 p1 = {0.5, 0, 0, 1};
    vector_float4 result = pattern_at(pattern, p1);
    
    vector_float4 expected = {0.5, 0.5, 0.5, 1};
    XCTAssertTrue([self vector:result equalsVector:expected tolerance:0.0001]);
}

// Chapter 10 - Testing pattern_at function with ring pattern
- (void)testPatternAtWithRingPattern {
    vector_float4 white = {1, 1, 1, 1};
    vector_float4 black = {0, 0, 0, 1};
    SurfacePattern pattern = pattern_ring(white, black);
    
    vector_float4 p1 = {0, 0, 0, 1};  // Center
    vector_float4 p2 = {1.5, 0, 0, 1}; // Distance 1.5, floor(1.5) = 1 (odd)
    
    XCTAssertTrue([self vector:pattern_at(pattern, p1) equalsVector:white tolerance:0.0001]);
    XCTAssertTrue([self vector:pattern_at(pattern, p2) equalsVector:black tolerance:0.0001]);
}

// Chapter 10 - Testing pattern_at function with checker pattern
- (void)testPatternAtWithCheckerPattern {
    vector_float4 white = {1, 1, 1, 1};
    vector_float4 black = {0, 0, 0, 1};
    SurfacePattern pattern = pattern_checker(white, black);
    
    vector_float4 p1 = {0, 0, 0, 1};   // 0+0+0=0 (even)
    vector_float4 p2 = {1, 1, 1, 1};   // 1+1+1=3 (odd)
    
    XCTAssertTrue([self vector:pattern_at(pattern, p1) equalsVector:white tolerance:0.0001]);
    XCTAssertTrue([self vector:pattern_at(pattern, p2) equalsVector:black tolerance:0.0001]);
}

// Chapter 10 - Pattern with identity transform should work the same
- (void)testPatternWithIdentityTransform {
    vector_float4 white = {1, 1, 1, 1};
    vector_float4 black = {0, 0, 0, 1};
    SurfacePattern pattern = pattern_stripe(white, black);
    // Identity transform by default
    
    vector_float4 p1 = {0, 0, 0, 1};
    vector_float4 p2 = {1, 0, 0, 1};
    
    // pattern_at applies the inverse transform, which for identity does nothing
    XCTAssertTrue([self vector:pattern_at(pattern, p1) equalsVector:white tolerance:0.0001]);
    XCTAssertTrue([self vector:pattern_at(pattern, p2) equalsVector:black tolerance:0.0001]);
}

// Chapter 10 - Pattern transformed with translation
- (void)testPatternWithTranslation {
    vector_float4 white = {1, 1, 1, 1};
    vector_float4 black = {0, 0, 0, 1};
    SurfacePattern pattern = pattern_stripe(white, black);
    
    Matrix4x4 trans = matrix_translation(0.5, 0, 0);
    pattern_set_transform(&pattern, trans);
    
    // Point at (0.5, 0, 0) in pattern space
    // After inverse transform (translate by -0.5), becomes (0, 0, 0)
    // 0 is in white stripe
    vector_float4 point = {0.5, 0, 0, 1};
    vector_float4 result = pattern_at(pattern, point);
    
    XCTAssertTrue([self vector:result equalsVector:white tolerance:0.0001]);
}

// Chapter 10 - Pattern transformed with rotation
- (void)testPatternWithRotation {
    vector_float4 white = {1, 1, 1, 1};
    vector_float4 black = {0, 0, 0, 1};
    SurfacePattern pattern = pattern_stripe(white, black);
    
    // Rotate stripes to run along Z axis instead of X
    Matrix4x4 rot = matrix_rotation_y(M_PI_2);
    pattern_set_transform(&pattern, rot);
    
    // Point at (0, 0, 1) in pattern space
    // After rotation, the point aligns with the stripe direction
    // This test verifies the pattern transform works
    vector_float4 point = {0, 0, 1, 1};
    vector_float4 result = pattern_at(pattern, point);
    
    // After Y rotation by 90 degrees, the point (0, 0, 1) becomes (1, 0, 0) approximately
    // So it should be in the black stripe (x=1)
    XCTAssertTrue([self vector:result equalsVector:black tolerance:0.0001]);
}

@end
