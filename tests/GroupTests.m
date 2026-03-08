#import <XCTest/XCTest.h>
#import "SharedTypes.h"

@interface GroupTests : XCTestCase
@end

@implementation GroupTests

// Helper to check if two vectors are approximately equal
- (BOOL)vector:(vector_float4)a equalsVector:(vector_float4)b tolerance:(float)tol {
    return simd_all(simd_abs(a - b) < tol);
}

// MARK: - Group Creation Tests

// Chapter 14 - Creating a new group
- (void)testCreatingGroup {
    Group g = group_create(1);
    
    XCTAssertEqual(g.id, 1);
    XCTAssertEqual(g.child_count, 0);
    
    // Check identity transform
    vector_float4 col0 = {1, 0, 0, 0};
    XCTAssertTrue([self vector:g.transform.columns[0] equalsVector:col0 tolerance:0.0001]);
    
    vector_float4 col1 = {0, 1, 0, 0};
    XCTAssertTrue([self vector:g.transform.columns[1] equalsVector:col1 tolerance:0.0001]);
    
    vector_float4 col2 = {0, 0, 1, 0};
    XCTAssertTrue([self vector:g.transform.columns[2] equalsVector:col2 tolerance:0.0001]);
    
    vector_float4 col3 = {0, 0, 0, 1};
    XCTAssertTrue([self vector:g.transform.columns[3] equalsVector:col3 tolerance:0.0001]);
}

// Chapter 14 - A group has a default material
- (void)testGroupDefaultMaterial {
    Group g = group_create(1);
    
    XCTAssertEqualWithAccuracy(g.material.ambient, 0.1f, 0.0001);
    XCTAssertEqualWithAccuracy(g.material.diffuse, 0.9f, 0.0001);
}

// MARK: - Group Transformation Tests

// Chapter 14 - Setting group transform
- (void)testSettingGroupTransform {
    Group g = group_create(1);
    
    Matrix4x4 trans = matrix_translation(2, 3, 4);
    group_set_transform(&g, trans);
    
    vector_float4 expected = {2, 3, 4, 1};
    XCTAssertTrue([self vector:g.transform.columns[3] equalsVector:expected tolerance:0.0001]);
}

// MARK: - Group Child Tests

// Chapter 14 - Adding a child to a group
- (void)testAddingChildToGroup {
    Group g = group_create(1);
    
    int result = group_add_child(&g, 2, SHAPE_SPHERE);
    XCTAssertEqual(result, 1);  // Success
    XCTAssertEqual(g.child_count, 1);
    XCTAssertEqual(g.child_ids[0], 2);
    XCTAssertEqual(g.child_types[0], SHAPE_SPHERE);
}

// Chapter 14 - Adding multiple children to a group
- (void)testAddingMultipleChildrenToGroup {
    Group g = group_create(1);
    
    group_add_child(&g, 2, SHAPE_SPHERE);
    group_add_child(&g, 3, SHAPE_CUBE);
    group_add_child(&g, 4, SHAPE_CYLINDER);
    
    XCTAssertEqual(g.child_count, 3);
    XCTAssertEqual(g.child_ids[0], 2);
    XCTAssertEqual(g.child_ids[1], 3);
    XCTAssertEqual(g.child_ids[2], 4);
}

// Chapter 14 - Group has limited capacity
- (void)testGroupCapacity {
    Group g = group_create(1);
    
    // Try to add more than MAX_GROUP_CHILDREN (20)
    int added = 0;
    for (int i = 0; i < 25; i++) {
        if (group_add_child(&g, i + 2, SHAPE_SPHERE)) {
            added++;
        }
    }
    
    XCTAssertEqual(added, 20);  // Should only add 20 (MAX_GROUP_CHILDREN)
    XCTAssertEqual(g.child_count, 20);
}

// MARK: - Shape Type Tests

// Chapter 14 - Shape type enumeration values
- (void)testShapeTypeEnumeration {
    XCTAssertEqual(SHAPE_SPHERE, 0);
    XCTAssertEqual(SHAPE_PLANE, 1);
    XCTAssertEqual(SHAPE_CUBE, 2);
    XCTAssertEqual(SHAPE_CYLINDER, 3);
    XCTAssertEqual(SHAPE_GROUP, 4);
}

// Chapter 14 - Adding different shape types to group
- (void)testAddingDifferentShapeTypes {
    Group g = group_create(1);
    
    group_add_child(&g, 10, SHAPE_SPHERE);
    group_add_child(&g, 20, SHAPE_PLANE);
    group_add_child(&g, 30, SHAPE_CUBE);
    group_add_child(&g, 40, SHAPE_CYLINDER);
    group_add_child(&g, 50, SHAPE_GROUP);
    
    XCTAssertEqual(g.child_types[0], SHAPE_SPHERE);
    XCTAssertEqual(g.child_types[1], SHAPE_PLANE);
    XCTAssertEqual(g.child_types[2], SHAPE_CUBE);
    XCTAssertEqual(g.child_types[3], SHAPE_CYLINDER);
    XCTAssertEqual(g.child_types[4], SHAPE_GROUP);
}

// MARK: - Group Identity

// Chapter 14 - Group identity is preserved
- (void)testGroupIdentityPreserved {
    Group g1 = group_create(1);
    Group g2 = group_create(2);
    
    XCTAssertEqual(g1.id, 1);
    XCTAssertEqual(g2.id, 2);
    XCTAssertNotEqual(g1.id, g2.id);
}

// Chapter 14 - Empty group has no children
- (void)testEmptyGroup {
    Group g = group_create(1);
    
    XCTAssertEqual(g.child_count, 0);
}

// MARK: - Hexagon Tests

// Chapter 14 - Hexagon corner creates a sphere with correct transform
- (void)testHexagonCorner {
    Sphere corner = hexagon_corner(100);
    
    XCTAssertEqual(corner.id, 100);
    
    // The corner should be scaled by 25% and translated -1 in z
    // Transform should be: translation(0, 0, -1) * scaling(0.25, 0.25, 0.25)
    // Which means: scale first (0.25), then translate (-1 in z)
    
    // After scaling 0.25 and translating -1 in z, the scale should be 0.25
    // and the translation z should be -1
    // But since transform combines them, we check the combined transform
    
    // For a point at origin after transform:
    // scale: (0, 0, 0) -> (0, 0, 0)
    // then translate: (0, 0, -1)
    vector_float4 origin = {0, 0, 0, 1};
    matrix_float4x4 mat = matrix_from_columns(corner.transform.columns[0], 
                                               corner.transform.columns[1],
                                               corner.transform.columns[2],
                                               corner.transform.columns[3]);
    vector_float4 transformed = matrix_multiply(mat, origin);
    
    // Should be at (0, 0, -1) after transform
    XCTAssertEqualWithAccuracy(transformed.x, 0.0f, 0.0001);
    XCTAssertEqualWithAccuracy(transformed.y, 0.0f, 0.0001);
    XCTAssertEqualWithAccuracy(transformed.z, -1.0f, 0.0001);
}

// Chapter 14 - Hexagon edge creates a cylinder with correct bounds
- (void)testHexagonEdge {
    Cylinder edge = hexagon_edge(101);
    
    XCTAssertEqual(edge.id, 101);
    
    // Cylinder should have bounds 0 to 1
    XCTAssertEqualWithAccuracy(edge.minimum, 0.0f, 0.0001);
    XCTAssertEqualWithAccuracy(edge.maximum, 1.0f, 0.0001);
    
    // Transform should exist (not identity)
    // Check that it's different from identity by checking one element
    XCTAssertTrue(edge.transform.columns[0].x != 1.0f || edge.transform.columns[3].z != 0.0f);
}

// Chapter 14 - Matrix multiply helper works correctly
- (void)testMatrix4x4Multiply {
    Matrix4x4 scale = matrix_scaling(2.0f, 2.0f, 2.0f);
    Matrix4x4 trans = matrix_translation(1.0f, 2.0f, 3.0f);
    
    // trans * scale means: scale first, then translate
    Matrix4x4 result = matrix4x4_multiply(trans, scale);
    
    // Check that scaling is applied
    // Column 0 should be (2, 0, 0, 0) - scaled x
    XCTAssertEqualWithAccuracy(result.columns[0].x, 2.0f, 0.0001);
    XCTAssertEqualWithAccuracy(result.columns[0].y, 0.0f, 0.0001);
    XCTAssertEqualWithAccuracy(result.columns[0].z, 0.0f, 0.0001);
    
    // Column 3 should have translation (1, 2, 3)
    XCTAssertEqualWithAccuracy(result.columns[3].x, 1.0f, 0.0001);
    XCTAssertEqualWithAccuracy(result.columns[3].y, 2.0f, 0.0001);
    XCTAssertEqualWithAccuracy(result.columns[3].z, 3.0f, 0.0001);
}

@end
