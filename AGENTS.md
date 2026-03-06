# AGENTS.md

Guidelines for agentic coding agents working in this Metal Ray Tracer repository.

## Project Overview

macOS Metal-based ray tracer implementing the "Ray Tracer Challenge" book. Uses Objective-C for app logic and Metal Shading Language for GPU compute kernels and rendering.

## Build Commands

```bash
# Configure build
cd build && cmake ..

# Build the project
make -C build

# Build Metal shaders only
make -C build MetalShaders

# Run the app
open build/RayTracer.app
# Or from CLI:
./build/RayTracer.app/Contents/MacOS/RayTracer

# Clean build
rm -rf build/* && cd build && cmake .. && make
```

## Testing

Tests use XCTest framework following "The Ray Tracer Challenge" TDD style.

```bash
# Build tests
make -C build RayTracerTests

# Run all tests
make -C build test
# Or:
./build/RayTracerTests.app/Contents/MacOS/RayTracerTests

# Run single test class
./build/RayTracerTests.app/Contents/MacOS/RayTracerTests -c TupleTests

# Run single test method
./build/RayTracerTests.app/Contents/MacOS/RayTracerTests -c TupleTests -m testTupleIsPoint
```

**Adding New Tests**:
- Add test files to `tests/` directory (e.g., `MatrixTests.m`)
- Follow naming: `[Feature]Tests.m` with test methods `test[Description]`
- Update `CMakeLists.txt` TEST_SOURCES if adding new files
- Tests follow Chapter X of the book - use `XCTAssertEqual` for exact, `XCTAssertEqualWithAccuracy` for floats

## Code Style Guidelines

### Objective-C

**Imports**: Use `#import` for Objective-C headers, `#include` for C/C++ headers
- System frameworks first: `#import <Cocoa/Cocoa.h>`
- Then project headers: `#import "AppDelegate.h"`

**Formatting**:
- 4 spaces for indentation (no tabs)
- Opening braces on same line for methods
- Line length: ~100 characters preferred

```objc
- (void)methodName {
    if (condition) {
        // code
    }
}
```

**Naming**:
- Classes: PascalCase (e.g., `AppDelegate`, `Renderer`)
- Methods: camelCase with full words (e.g., `initWithMetalKitView:`)
- Instance variables: prefix with underscore (e.g., `_device`, `_commandQueue`)
- Constants: UPPER_SNAKE_CASE with defines (e.g., `TUPLE_POINT`)

**Properties**:
```objc
@property (strong, nonatomic) NSWindow *window;
@property (weak, readonly) id<MTLDevice> device;
```

**Nullability**: Use nullability annotations
```objc
- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)mtkView;
```

**Error Handling**: Check Metal API errors, log with NSLog
```objc
if (!self.view.device) {
    NSLog(@"Metal is not supported on this device");
    return;
}
```

### Metal Shading Language

**Style**:
- Use `[[attribute]]` syntax for shader attributes
- Buffer indices: explicit numbering `[[buffer(0)]]`
- Kernel functions use descriptive names: `tuple_add`, `vector_normalize`
- Include guards for shared headers between Metal and C/ObjC

```metal
kernel void tuple_add(device const Tuple* a [[buffer(0)]],
                      device const Tuple* b [[buffer(1)]],
                      device Tuple* result [[buffer(2)]],
                      uint index [[thread_position_in_grid]])
{
    result[index].components = a[index].components + b[index].components;
}
```

### C/C++ Headers (SharedTypes.h)

- Use `#ifndef`/`#define`/`#endif` include guards
- Share structs between CPU and GPU code
- Use `vector_float4` from `simd/simd.h` for SIMD types

## Project Structure

```
.
├── CMakeLists.txt          # Build configuration
├── src/                    # Objective-C source files
│   ├── main.m             # Entry point
│   ├── AppDelegate.{h,m}  # App lifecycle
│   ├── Renderer.{h,m}     # Metal rendering
│   └── SharedTypes.h      # Shared CPU/GPU types
├── shaders/               # Metal shading language
│   └── Shaders.metal      # Kernels and shaders
├── tests/                 # XCTest unit tests
│   ├── TupleTests.m       # Chapter 1: Tuples
│   ├── VectorTests.m      # Chapter 1: Vectors
│   └── main.m             # Test entry point
└── build/                 # Build output (generated)
    └── RayTracer.app      # macOS app bundle
```

## Architecture Notes

- **AppDelegate**: Sets up NSWindow and MTKView, owns Renderer
- **Renderer**: Implements MTKViewDelegate, manages Metal pipeline
- **Shaders.metal**: Contains vertex/fragment shaders and compute kernels
- Uses full-screen quad for ray tracing output (vertex shader passes through)
- Compute kernels operate on Tuple structs for vector math operations

## Adding New Features

1. **New compute kernel**: Add to `shaders/Shaders.metal`, follow existing naming
2. **New UI**: Modify `AppDelegate.m`, maintain 800x600 default window size
3. **Shared types**: Add to `SharedTypes.h` with proper alignment for GPU
4. **Metal resources**: Ensure proper error checking and nil handling

## Dependencies

- macOS SDK (Metal, MetalKit, Cocoa, QuartzCore)
- CMake 3.15+
- Xcode command line tools (for `xcrun metal` and `xcrun metallib`)

## Important Constraints

- Metal code requires macOS device with Metal support
- Shader compilation happens at build time, not runtime
- App bundle structure required for resource loading
- No dynamic library loading - everything linked statically
