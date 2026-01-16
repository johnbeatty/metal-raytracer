#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "SharedTypes.h"

// Helper to compare floats
BOOL float_equal(float a, float b) {
    return fabs(a - b) < 0.0001;
}

// Helper to print tuples
NSString* tuple_string(Tuple t) {
    return [NSString stringWithFormat:@"(%.2f, %.2f, %.2f, %.2f)", 
            t.components.x, t.components.y, t.components.z, t.components.w];
}

// Test Runner Class
@interface TestRunner : NSObject
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLLibrary> library;

- (instancetype)init;
- (void)runTests;
@end

@implementation TestRunner

- (instancetype)init {
    self = [super init];
    if (self) {
        _device = MTLCreateSystemDefaultDevice();
        _commandQueue = [_device newCommandQueue];
        
        NSError *error = nil;
        // Try local file first (CLI build)
        NSURL *libraryURL = [[NSURL fileURLWithPath:@"default.metallib"] absoluteURL];
        _library = [_device newLibraryWithURL:libraryURL error:&error];
        
        if (!_library) {
            NSLog(@"Failed to load library: %@", error);
        }
    }
    return self;
}

- (void)dispatchKernel:(NSString *)functionName 
              buffers:(NSArray<id<MTLBuffer>> *)buffers 
           itemCount:(NSUInteger)count {
    
    id<MTLFunction> function = [_library newFunctionWithName:functionName];
    if (!function) {
        NSLog(@"Error: Could not find function %@", functionName);
        return;
    }
    
    NSError *error = nil;
    id<MTLComputePipelineState> pso = [_device newComputePipelineStateWithFunction:function error:&error];
    if (!pso) {
        NSLog(@"Error creating PSO: %@", error);
        return;
    }
    
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pso];
    
    for (NSUInteger i = 0; i < buffers.count; i++) {
        [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    }
    
    MTLSize gridSize = MTLSizeMake(count, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(MIN(count, pso.maxTotalThreadsPerThreadgroup), 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

// Tests

- (void)testTupleAddition {
    NSLog(@"--- Testing Tuple Addition ---");
    
    Tuple a = { .components = {3, -2, 5, 1} };
    Tuple b = { .components = {-2, 3, 1, 0} };
    Tuple expected = { .components = {1, 1, 6, 1} };
    
    id<MTLBuffer> bufA = [_device newBufferWithBytes:&a length:sizeof(Tuple) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [_device newBufferWithBytes:&b length:sizeof(Tuple) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufRes = [_device newBufferWithLength:sizeof(Tuple) options:MTLResourceStorageModeShared];
    
    [self dispatchKernel:@"tuple_add" buffers:@[bufA, bufB, bufRes] itemCount:1];
    
    Tuple *result = (Tuple *)bufRes.contents;
    NSLog(@"%@ + %@ = %@", tuple_string(a), tuple_string(b), tuple_string(*result));
    
    if (float_equal(result->components.x, expected.components.x) &&
        float_equal(result->components.y, expected.components.y) &&
        float_equal(result->components.z, expected.components.z) &&
        float_equal(result->components.w, expected.components.w)) {
        NSLog(@"✅ PASS");
    } else {
        NSLog(@"❌ FAIL. Expected %@", tuple_string(expected));
    }
}

- (void)testTupleSubtraction {
    NSLog(@"--- Testing Tuple Subtraction (Point - Point = Vector) ---");
    
    Tuple a = { .components = {3, 2, 1, 1} };
    Tuple b = { .components = {5, 6, 7, 1} };
    Tuple expected = { .components = {-2, -4, -6, 0} };
    
    id<MTLBuffer> bufA = [_device newBufferWithBytes:&a length:sizeof(Tuple) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [_device newBufferWithBytes:&b length:sizeof(Tuple) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufRes = [_device newBufferWithLength:sizeof(Tuple) options:MTLResourceStorageModeShared];
    
    [self dispatchKernel:@"tuple_subtract" buffers:@[bufA, bufB, bufRes] itemCount:1];
    
    Tuple *result = (Tuple *)bufRes.contents;
    NSLog(@"%@ - %@ = %@", tuple_string(a), tuple_string(b), tuple_string(*result));
    
    if (float_equal(result->components.x, expected.components.x) &&
        float_equal(result->components.y, expected.components.y) &&
        float_equal(result->components.z, expected.components.z) &&
        float_equal(result->components.w, expected.components.w)) {
        NSLog(@"✅ PASS");
    } else {
        NSLog(@"❌ FAIL. Expected %@", tuple_string(expected));
    }
}

- (void)testTupleNegation {
    NSLog(@"--- Testing Tuple Negation ---");
    
    Tuple a = { .components = {1, -2, 3, -4} };
    Tuple expected = { .components = {-1, 2, -3, 4} };
    
    id<MTLBuffer> bufA = [_device newBufferWithBytes:&a length:sizeof(Tuple) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufRes = [_device newBufferWithLength:sizeof(Tuple) options:MTLResourceStorageModeShared];
    
    [self dispatchKernel:@"tuple_negate" buffers:@[bufA, bufRes] itemCount:1];
    
    Tuple *result = (Tuple *)bufRes.contents;
    NSLog(@"-%@ = %@", tuple_string(a), tuple_string(*result));
    
    if (float_equal(result->components.x, expected.components.x) &&
        float_equal(result->components.y, expected.components.y) &&
        float_equal(result->components.z, expected.components.z) &&
        float_equal(result->components.w, expected.components.w)) {
        NSLog(@"✅ PASS");
    } else {
        NSLog(@"❌ FAIL. Expected %@", tuple_string(expected));
    }
}

- (void)testScalarMultiplication {
    NSLog(@"--- Testing Scalar Multiplication ---");
    
    Tuple a = { .components = {1, -2, 3, -4} };
    float scalar = 3.5;
    Tuple expected = { .components = {3.5, -7, 10.5, -14} };
    
    id<MTLBuffer> bufA = [_device newBufferWithBytes:&a length:sizeof(Tuple) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufScalar = [_device newBufferWithBytes:&scalar length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufRes = [_device newBufferWithLength:sizeof(Tuple) options:MTLResourceStorageModeShared];
    
    [self dispatchKernel:@"tuple_multiply_scalar" buffers:@[bufA, bufScalar, bufRes] itemCount:1];
    
    Tuple *result = (Tuple *)bufRes.contents;
    NSLog(@"%@ * %.2f = %@", tuple_string(a), scalar, tuple_string(*result));
    
    if (float_equal(result->components.x, expected.components.x) &&
        float_equal(result->components.y, expected.components.y) &&
        float_equal(result->components.z, expected.components.z) &&
        float_equal(result->components.w, expected.components.w)) {
        NSLog(@"✅ PASS");
    } else {
        NSLog(@"❌ FAIL. Expected %@", tuple_string(expected));
    }
}

- (void)testScalarDivision {
    NSLog(@"--- Testing Scalar Division ---");
    
    Tuple a = { .components = {1, -2, 3, -4} };
    float scalar = 2.0;
    Tuple expected = { .components = {0.5, -1, 1.5, -2} };
    
    id<MTLBuffer> bufA = [_device newBufferWithBytes:&a length:sizeof(Tuple) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufScalar = [_device newBufferWithBytes:&scalar length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufRes = [_device newBufferWithLength:sizeof(Tuple) options:MTLResourceStorageModeShared];
    
    [self dispatchKernel:@"tuple_divide_scalar" buffers:@[bufA, bufScalar, bufRes] itemCount:1];
    
    Tuple *result = (Tuple *)bufRes.contents;
    NSLog(@"%@ / %.2f = %@", tuple_string(a), scalar, tuple_string(*result));
    
    if (float_equal(result->components.x, expected.components.x) &&
        float_equal(result->components.y, expected.components.y) &&
        float_equal(result->components.z, expected.components.z) &&
        float_equal(result->components.w, expected.components.w)) {
        NSLog(@"✅ PASS");
    } else {
        NSLog(@"❌ FAIL. Expected %@", tuple_string(expected));
    }
}

- (void)testVectorMagnitude {
    NSLog(@"--- Testing Vector Magnitude ---");
    
    Tuple a = { .components = {-1, -2, -3, 0} };
    float expected = sqrtf(14);
    
    id<MTLBuffer> bufA = [_device newBufferWithBytes:&a length:sizeof(Tuple) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufRes = [_device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    
    [self dispatchKernel:@"vector_magnitude" buffers:@[bufA, bufRes] itemCount:1];
    
    float *result = (float *)bufRes.contents;
    NSLog(@"magnitude(%@) = %.4f", tuple_string(a), *result);
    
    if (float_equal(*result, expected)) {
        NSLog(@"✅ PASS");
    } else {
        NSLog(@"❌ FAIL. Expected %.4f", expected);
    }
}

- (void)testVectorNormalize {
    NSLog(@"--- Testing Vector Normalize ---");
    
    Tuple a = { .components = {4, 0, 0, 0} };
    Tuple expected = { .components = {1, 0, 0, 0} };
    
    id<MTLBuffer> bufA = [_device newBufferWithBytes:&a length:sizeof(Tuple) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufRes = [_device newBufferWithLength:sizeof(Tuple) options:MTLResourceStorageModeShared];
    
    [self dispatchKernel:@"vector_normalize" buffers:@[bufA, bufRes] itemCount:1];
    
    Tuple *result = (Tuple *)bufRes.contents;
    NSLog(@"normalize(%@) = %@", tuple_string(a), tuple_string(*result));
    
    if (float_equal(result->components.x, expected.components.x) &&
        float_equal(result->components.y, expected.components.y) &&
        float_equal(result->components.z, expected.components.z) &&
        float_equal(result->components.w, expected.components.w)) {
        NSLog(@"✅ PASS");
    } else {
        NSLog(@"❌ FAIL. Expected %@", tuple_string(expected));
    }
}

- (void)testVectorDot {
    NSLog(@"--- Testing Dot Product ---");
    
    Tuple a = { .components = {1, 2, 3, 0} };
    Tuple b = { .components = {2, 3, 4, 0} };
    float expected = 20.0;
    
    id<MTLBuffer> bufA = [_device newBufferWithBytes:&a length:sizeof(Tuple) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [_device newBufferWithBytes:&b length:sizeof(Tuple) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufRes = [_device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    
    [self dispatchKernel:@"vector_dot" buffers:@[bufA, bufB, bufRes] itemCount:1];
    
    float *result = (float *)bufRes.contents;
    NSLog(@"dot(%@, %@) = %.2f", tuple_string(a), tuple_string(b), *result);
    
    if (float_equal(*result, expected)) {
        NSLog(@"✅ PASS");
    } else {
        NSLog(@"❌ FAIL. Expected %.2f", expected);
    }
}

- (void)testVectorCross {
    NSLog(@"--- Testing Cross Product ---");
    
    Tuple a = { .components = {1, 2, 3, 0} };
    Tuple b = { .components = {2, 3, 4, 0} };
    Tuple expected = { .components = {-1, 2, -1, 0} };
    
    id<MTLBuffer> bufA = [_device newBufferWithBytes:&a length:sizeof(Tuple) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [_device newBufferWithBytes:&b length:sizeof(Tuple) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufRes = [_device newBufferWithLength:sizeof(Tuple) options:MTLResourceStorageModeShared];
    
    [self dispatchKernel:@"vector_cross" buffers:@[bufA, bufB, bufRes] itemCount:1];
    
    Tuple *result = (Tuple *)bufRes.contents;
    NSLog(@"cross(%@, %@) = %@", tuple_string(a), tuple_string(b), tuple_string(*result));
    
    if (float_equal(result->components.x, expected.components.x) &&
        float_equal(result->components.y, expected.components.y) &&
        float_equal(result->components.z, expected.components.z) &&
        float_equal(result->components.w, expected.components.w)) {
        NSLog(@"✅ PASS");
    } else {
        NSLog(@"❌ FAIL. Expected %@", tuple_string(expected));
    }
}

- (void)runTests {
    [self testTupleAddition];
    [self testTupleSubtraction];
    [self testTupleNegation];
    [self testScalarMultiplication];
    [self testScalarDivision];
    [self testVectorMagnitude];
    [self testVectorNormalize];
    [self testVectorDot];
    [self testVectorCross];
}

@end

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        TestRunner *runner = [[TestRunner alloc] init];
        [runner runTests];
    }
    return 0;
}
