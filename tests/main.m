// Test runner for RayTracer tests
// Uses XCTestObservation to run and report tests

#import <XCTest/XCTest.h>
#import <objc/runtime.h>

@interface TestObserver : NSObject <XCTestObservation>
@property (nonatomic, assign) int failureCount;
@end

@implementation TestObserver

- (instancetype)init {
    self = [super init];
    if (self) {
        _failureCount = 0;
    }
    return self;
}

- (void)testSuiteWillStart:(XCTestSuite *)testSuite {
    // Suite starting
}

- (void)testCaseDidFail:(XCTestCase *)testCase withDescription:(NSString *)description inFile:(NSString *)filePath atLine:(NSUInteger)lineNumber {
    _failureCount++;
    NSLog(@"FAIL: %@", description);
}

- (void)testSuiteDidFinish:(XCTestSuite *)testSuite {
    // Suite finished
}

@end

int main(int argc, char *argv[]) {
    @autoreleasepool {
        // Register our observer
        TestObserver *observer = [[TestObserver alloc] init];
        [[XCTestObservationCenter sharedTestObservationCenter] addTestObserver:observer];
        
        // Create a test suite with all tests
        XCTestSuite *suite = [XCTestSuite defaultTestSuite];
        
        // Run the tests
        [suite runTest];
        
        return observer.failureCount;
    }
}
