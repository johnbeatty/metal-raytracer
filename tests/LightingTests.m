#import <XCTest/XCTest.h>
#import "SharedTypes.h"

@interface LightingTests : XCTestCase
@end

@implementation LightingTests

// Helper to check if two vectors are approximately equal
- (BOOL)vector:(vector_float4)a equalsVector:(vector_float4)b tolerance:(float)tol {
    return simd_all(simd_abs(a - b) < tol);
}

// Chapter 6 - The normal on a sphere at a point on the X axis
- (void)testNormalOnSphereAtXAxis {
    Sphere s = sphere_create(1);
    vector_float4 p = {1, 0, 0, 1};
    vector_float4 n = sphere_normal_at(s, p);
    vector_float4 expected = {1, 0, 0, 0};
    
    XCTAssertTrue([self vector:n equalsVector:expected tolerance:0.0001]);
}

// Chapter 6 - The normal on a sphere at a point on the Y axis
- (void)testNormalOnSphereAtYAxis {
    Sphere s = sphere_create(1);
    vector_float4 p = {0, 1, 0, 1};
    vector_float4 n = sphere_normal_at(s, p);
    vector_float4 expected = {0, 1, 0, 0};
    
    XCTAssertTrue([self vector:n equalsVector:expected tolerance:0.0001]);
}

// Chapter 6 - The normal on a sphere at a point on the Z axis
- (void)testNormalOnSphereAtZAxis {
    Sphere s = sphere_create(1);
    vector_float4 p = {0, 0, 1, 1};
    vector_float4 n = sphere_normal_at(s, p);
    vector_float4 expected = {0, 0, 1, 0};
    
    XCTAssertTrue([self vector:n equalsVector:expected tolerance:0.0001]);
}

// Chapter 6 - The normal on a sphere at a non-axial point
- (void)testNormalOnSphereAtNonAxialPoint {
    Sphere s = sphere_create(1);
    float v = sqrtf(3.0) / 3.0;
    vector_float4 p = {v, v, v, 1};
    vector_float4 n = sphere_normal_at(s, p);
    vector_float4 expected = {v, v, v, 0};
    
    XCTAssertTrue([self vector:n equalsVector:expected tolerance:0.0001]);
    // Normal should be normalized
    XCTAssertEqualWithAccuracy(simd_length(n.xyz), 1.0, 0.0001);
}

// Chapter 6 - Computing the normal on a translated sphere
- (void)testNormalOnTranslatedSphere {
    Sphere s = sphere_create(1);
    Matrix4x4 transform = matrix_translation(0, 1, 0);
    sphere_set_transform(&s, transform);
    
    vector_float4 p = {0, 1.70711, -0.70711, 1};
    vector_float4 n = sphere_normal_at(s, p);
    vector_float4 expected = {0, 0.70711, -0.70711, 0};
    
    XCTAssertTrue([self vector:n equalsVector:expected tolerance:0.0001]);
}

// Chapter 6 - Computing the normal on a transformed sphere
- (void)testNormalOnTransformedSphere {
    Sphere s = sphere_create(1);
    Matrix4x4 scale = matrix_scaling(1, 0.5, 1);
    Matrix4x4 rotate = matrix_rotation_z(M_PI / 5.0);
    Matrix4x4 m = [self multiplyMatrix:scale byMatrix:rotate];
    sphere_set_transform(&s, m);
    
    float v = sqrtf(2.0) / 2.0;
    vector_float4 p = {0, v, -v, 1};
    vector_float4 n = sphere_normal_at(s, p);
    vector_float4 expected = {0, 0.97014, -0.24254, 0};
    
    XCTAssertTrue([self vector:n equalsVector:expected tolerance:0.0001]);
}

// Helper to multiply two matrices
- (Matrix4x4)multiplyMatrix:(Matrix4x4)a byMatrix:(Matrix4x4)b {
    matrix_float4x4 ma = matrix_from_columns(a.columns[0], a.columns[1], a.columns[2], a.columns[3]);
    matrix_float4x4 mb = matrix_from_columns(b.columns[0], b.columns[1], b.columns[2], b.columns[3]);
    matrix_float4x4 mc = matrix_multiply(ma, mb);
    Matrix4x4 result;
    result.columns[0] = mc.columns[0];
    result.columns[1] = mc.columns[1];
    result.columns[2] = mc.columns[2];
    result.columns[3] = mc.columns[3];
    return result;
}

// Chapter 6 - A point light has a position and intensity
- (void)testPointLightHasPositionAndIntensity {
    vector_float4 intensity = {1, 1, 1, 1};
    vector_float4 position = {0, 0, 0, 1};
    PointLight light = point_light_create(position, intensity);
    
    XCTAssertTrue([self vector:light.position equalsVector:position tolerance:0.0001]);
    XCTAssertTrue([self vector:light.intensity equalsVector:intensity tolerance:0.0001]);
}

// Chapter 6 - The default material
- (void)testDefaultMaterial {
    Material m = DEFAULT_MATERIAL;
    vector_float4 expected_color = {1, 1, 1, 1};
    
    XCTAssertTrue([self vector:m.color equalsVector:expected_color tolerance:0.0001]);
    XCTAssertEqualWithAccuracy(m.ambient, 0.1, 0.0001);
    XCTAssertEqualWithAccuracy(m.diffuse, 0.9, 0.0001);
    XCTAssertEqualWithAccuracy(m.specular, 0.9, 0.0001);
    XCTAssertEqualWithAccuracy(m.shininess, 200.0, 0.0001);
}

// Chapter 6 - Reflecting a vector approaching at 45 degrees
- (void)testReflectingVectorApproaching45Degrees {
    vector_float4 v = {1, -1, 0, 0};
    vector_float4 n = {0, 1, 0, 0};
    vector_float4 r = [self reflect:v normal:n];
    vector_float4 expected = {1, 1, 0, 0};
    
    XCTAssertTrue([self vector:r equalsVector:expected tolerance:0.0001]);
}

// Chapter 6 - Reflecting a vector off a slanted surface
- (void)testReflectingVectorOffSlantedSurface {
    vector_float4 v = {0, -1, 0, 0};
    float v_val = sqrtf(2.0) / 2.0;
    vector_float4 n = {v_val, v_val, 0, 0};
    vector_float4 r = [self reflect:v normal:n];
    vector_float4 expected = {1, 0, 0, 0};
    
    XCTAssertTrue([self vector:r equalsVector:expected tolerance:0.0001]);
}

// Helper: Reflect vector around normal
- (vector_float4)reflect:(vector_float4)v normal:(vector_float4)n {
    return v - n * 2.0 * simd_dot(v, n);
}

// Chapter 6 - Lighting with eye between light and surface
- (void)testLightingWithEyeBetweenLightAndSurface {
    Material m = DEFAULT_MATERIAL;
    vector_float4 position = {0, 0, 0, 1};
    
    vector_float4 eye_vector = {0, 0, -1, 0};
    vector_float4 normal = {0, 0, -1, 0};
    PointLight light = point_light_create((vector_float4){0, 0, -10, 1}, 
                                          (vector_float4){1, 1, 1, 1});
    
    vector_float4 result = lighting(m, light, position, eye_vector, normal);
    vector_float4 expected = {1.9, 1.9, 1.9, 1};
    
    XCTAssertTrue([self vector:result equalsVector:expected tolerance:0.0001]);
}

// Chapter 6 - Lighting with eye between light and surface, eye offset 45 degrees
- (void)testLightingWithEyeOffset45Degrees {
    Material m = DEFAULT_MATERIAL;
    vector_float4 position = {0, 0, 0, 1};
    
    float v = sqrtf(2.0) / 2.0;
    vector_float4 eye_vector = {0, v, -v, 0};
    vector_float4 normal = {0, 0, -1, 0};
    PointLight light = point_light_create((vector_float4){0, 0, -10, 1}, 
                                          (vector_float4){1, 1, 1, 1});
    
    vector_float4 result = lighting(m, light, position, eye_vector, normal);
    vector_float4 expected = {1.0, 1.0, 1.0, 1};
    
    XCTAssertTrue([self vector:result equalsVector:expected tolerance:0.0001]);
}

// Chapter 6 - Lighting with eye opposite surface, light offset 45 degrees
- (void)testLightingWithLightOffset45Degrees {
    Material m = DEFAULT_MATERIAL;
    vector_float4 position = {0, 0, 0, 1};
    
    vector_float4 eye_vector = {0, 0, -1, 0};
    vector_float4 normal = {0, 0, -1, 0};
    PointLight light = point_light_create((vector_float4){0, 10, -10, 1}, 
                                          (vector_float4){1, 1, 1, 1});
    
    vector_float4 result = lighting(m, light, position, eye_vector, normal);
    vector_float4 expected = {0.7364, 0.7364, 0.7364, 1};
    
    XCTAssertTrue([self vector:result equalsVector:expected tolerance:0.0001]);
}

// Chapter 6 - Lighting with eye in the path of reflection vector
- (void)testLightingWithEyeInReflectionPath {
    Material m = DEFAULT_MATERIAL;
    vector_float4 position = {0, 0, 0, 1};
    
    float v = sqrtf(2.0) / 2.0;
    vector_float4 eye_vector = {0, -v, -v, 0};
    vector_float4 normal = {0, 0, -1, 0};
    PointLight light = point_light_create((vector_float4){0, 10, -10, 1}, 
                                          (vector_float4){1, 1, 1, 1});
    
    vector_float4 result = lighting(m, light, position, eye_vector, normal);
    vector_float4 expected = {1.6364, 1.6364, 1.6364, 1};
    
    XCTAssertTrue([self vector:result equalsVector:expected tolerance:0.0001]);
}

// Chapter 6 - Lighting with light behind the surface
- (void)testLightingWithLightBehindSurface {
    Material m = DEFAULT_MATERIAL;
    vector_float4 position = {0, 0, 0, 1};
    
    vector_float4 eye_vector = {0, 0, -1, 0};
    vector_float4 normal = {0, 0, -1, 0};
    PointLight light = point_light_create((vector_float4){0, 0, 10, 1}, 
                                          (vector_float4){1, 1, 1, 1});
    
    vector_float4 result = lighting(m, light, position, eye_vector, normal);
    vector_float4 expected = {0.1, 0.1, 0.1, 1};  // Only ambient
    
    XCTAssertTrue([self vector:result equalsVector:expected tolerance:0.0001]);
}

@end
