# AGENTS.md

Guidelines for agentic coding agents working in this Metal Ray Tracer repository.

## Project Overview

macOS Metal-based ray tracer implementing the "Ray Tracer Challenge" book. Uses Objective-C for app logic and Metal Shading Language for GPU compute kernels and rendering.

## Discoveries

- Metal doesn't have built-in `inverse()` or `determinant()` for 4x4 matrices - we implemented manual `matrix_inverse_4x4()` using cofactor expansion
- Texture caching issues: The metallib file must be explicitly copied to the app bundle's Resources directory after each build
- Compute shaders need forward declarations when functions call each other
- `half` is a reserved keyword in Metal - use `half_size` instead
- For top-down camera views, simple projection math is more reliable than full view transforms

## Accomplished

**Completed Chapters:**
1. **Chapter 1-2**: Tuples, vectors, points - 24 tests (TupleTests.m, VectorTests.m)
2. **Chapter 3**: Matrix operations (multiply, transpose, inverse) - 10 tests (MatrixTests.m)
3. **Chapter 4**: Transformations (translation, scaling, rotation X/Y/Z, shearing) - 19 tests (TransformationTests.m)
4. **Chapter 5**: Ray-sphere intersections, hit determination - 14 tests (IntersectionTests.m)
5. **Chapter 6**: Phong lighting (ambient, diffuse, specular) - 15 tests (LightingTests.m)
6. **Chapter 7**: World system, camera, multiple objects - 12 tests (WorldTests.m)
7. **Chapter 8**: Shadow detection - 7 tests (ShadowTests.m)
8. **Chapter 9**: Planes, hexagonal room demo - 8 tests (PlaneTests.m)
9. **Chapter 10**: Patterns (stripes, gradient, ring, checker) - 19 tests (PatternTests.m)

**Visual Progress:**
- Started with simple red/green gradient (Chapter 5-6)
- Added 3D shaded magenta sphere (Chapter 6)
- Full 6-sphere scene with shadows (Chapter 7-8)
- Hexagonal room with wood floor, viewed from above (Chapter 9)
- Four spheres with different patterns: stripes, gradient, rings, checkers (Chapter 10)

**Total: 128 tests passing** ✓

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
├── .gitignore              # Excludes build/, generated files
├── AGENTS.md               # Documentation for agents (build commands, code style)
├── src/                    # Objective-C source files
│   ├── main.m             # Entry point
│   ├── AppDelegate.{h,m}  # App lifecycle
│   ├── Renderer.{h,m}     # Metal rendering: compute pipeline, texture management
│   └── SharedTypes.h      # Shared CPU/GPU types
├── shaders/               # Metal shading language
│   └── Shaders.metal      # Kernels and shaders
├── tests/                 # XCTest unit tests
│   ├── TupleTests.m       # Chapter 1: Tuples
│   ├── VectorTests.m      # Chapter 1: Vectors
│   ├── MatrixTests.m      # Chapter 3: Matrices
│   ├── TransformationTests.m # Chapter 4: Transformations
│   ├── IntersectionTests.m # Chapter 5: Ray-sphere intersections
│   ├── LightingTests.m    # Chapter 6: Phong lighting
│   ├── WorldTests.m       # Chapter 7: World system
│   ├── ShadowTests.m      # Chapter 8: Shadows
│   ├── PlaneTests.m       # Chapter 9: Planes
│   └── main.m             # Test entry point
└── build/                 # Build output (generated)
    └── RayTracer.app      # macOS app bundle
```

**Key Technical Details:**
- **Metal Library**: `default.metallib` must be in `RayTracer.app/Contents/Resources/`
- **Resolution**: 1920x1080 Full HD
- **Compute Threads**: 16x16 threadgroups, dispatched for full resolution
- **SharedTypes.h**: Contains both CPU and GPU-compatible types, uses `#ifdef __METAL_VERSION__` for conditional compilation
- **Tests**: Run with `./build/RayTracerTests.app/Contents/MacOS/RayTracerTests`
- **App**: Run with `open ./build/RayTracer.app`

## Architecture Notes

- **AppDelegate**: Sets up NSWindow and MTKView, owns Renderer
- **Renderer**: Implements MTKViewDelegate, manages Metal pipeline, ensures metallib is copied to Resources
- **Shaders.metal**: Contains vertex/fragment shaders and compute kernels
- **SharedTypes.h**: Contains all math structures (Tuple, Matrix4x4, Ray, Sphere, Plane, Material, World, Camera) and lighting functions
- Uses full-screen quad for ray tracing output (vertex shader passes through)
- Compute kernels operate on Tuple structs for vector math operations
- Resolution: 1920x1080 Full HD
- Threadgroups: 16x16 dispatched for full resolution

## Adding New Features

1. **New compute kernel**: Add to `shaders/Shaders.metal`, follow existing naming
2. **New UI**: Modify `AppDelegate.m`, maintain 1920x1080 Full HD resolution
3. **Shared types**: Add to `SharedTypes.h` with proper alignment for GPU
4. **Metal resources**: Ensure proper error checking and nil handling

## Next Chapters Available

- Chapter 11: Reflection & Refraction (mirrors, glass)
- Chapter 12: Cubes
- Chapter 13: Cylinders
- Chapter 14: Groups (object hierarchies)

## Current Working State

- All 128 tests passing
- Four spheres with different patterns render correctly (stripes, gradient, rings, checkers)
- Ready to continue with Chapter 11 or user-directed improvements

## Dependencies

- macOS SDK (Metal, MetalKit, Cocoa, QuartzCore)
- CMake 3.15+
- Xcode command line tools (for `xcrun metal` and `xcrun metallib`)

## Important Constraints

- Metal code requires macOS device with Metal support
- Shader compilation happens at build time, not runtime
- App bundle structure required for resource loading
- No dynamic library loading - everything linked statically
