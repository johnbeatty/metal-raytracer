#import "Renderer.h"
#import "SharedTypes.h"

@implementation Renderer
{
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    id<MTLRenderPipelineState> _pipelineState;
    id<MTLBuffer> _vertexBuffer;
    NSUInteger _numVertices;
    vector_uint2 _viewportSize;
    
    // Chapter 5: Sphere silhouette rendering
    id<MTLComputePipelineState> _computePipelineState;
    id<MTLTexture> _sphereTexture;
    
    // Chapter 11: Camera animation
    id<MTLBuffer> _timeBuffer;
    float _time;
}

    - (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)mtkView
{
    self = [super init];
    if(self)
    {
        _device = mtkView.device;
        _commandQueue = [_device newCommandQueue];
        
        // Chapter 11: Initialize time tracking
        _time = 0.0f;
        _timeBuffer = [_device newBufferWithLength:sizeof(float) 
                                           options:MTLResourceStorageModeShared];
        
        [self loadMetal:mtkView];
        [self buildMesh];
    }
    return self;
}

- (void)loadMetal:(nonnull MTKView *)mtkView
{
    mtkView.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    
    NSError *error = nil;
    
    // Load local library (bundle or file)
    id<MTLLibrary> defaultLibrary = nil;
    NSURL *libraryURL = [[NSBundle mainBundle] URLForResource:@"default" withExtension:@"metallib"];
    
    if (libraryURL) {
        defaultLibrary = [_device newLibraryWithURL:libraryURL error:&error];
    } else {
        // Fallback for CLI/local testing if not correctly bundled
        NSURL *fallbackURL = [NSURL fileURLWithPath:@"default.metallib"];
        defaultLibrary = [_device newLibraryWithURL:fallbackURL error:&error];
    }
    
    if (!defaultLibrary) {
        NSLog(@"Failed to load library: %@", error);
        return;
    }
    
    id<MTLFunction> vertexFunction = [defaultLibrary newFunctionWithName:@"vertexShader"];
    id<MTLFunction> fragmentFunction = [defaultLibrary newFunctionWithName:@"fragmentShader"];
    
    MTLRenderPipelineDescriptor *pipelineStateDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineStateDescriptor.label = @"Simple Pipeline";
    pipelineStateDescriptor.vertexFunction = vertexFunction;
    pipelineStateDescriptor.fragmentFunction = fragmentFunction;
    pipelineStateDescriptor.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat;
    
    _pipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineStateDescriptor
                                                            error:&error];
    if (!_pipelineState) {
        NSLog(@"Failed to created pipeline state, error %@", error);
    }
    
    // Chapter 14: Create compute pipeline for hexagon demo
    id<MTLFunction> computeFunction = [defaultLibrary newFunctionWithName:@"render_hexagon_demo"];
    if (computeFunction) {
        NSLog(@"Found compute function 'render_hexagon_demo'");
        _computePipelineState = [_device newComputePipelineStateWithFunction:computeFunction error:&error];
        if (_computePipelineState) {
            NSLog(@"Successfully created compute pipeline");
        } else {
            NSLog(@"Failed to create compute pipeline state: %@", error);
        }
    } else {
        NSLog(@"Failed to find compute function 'render_hexagon_demo'");
    }
    
    // Chapter 5: Create texture for sphere rendering
    MTLTextureDescriptor *textureDescriptor = [[MTLTextureDescriptor alloc] init];
    textureDescriptor.pixelFormat = MTLPixelFormatRGBA8Unorm;
    textureDescriptor.width = 1920;  // Full HD resolution (was 400)
    textureDescriptor.height = 1080;
    textureDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    _sphereTexture = [_device newTextureWithDescriptor:textureDescriptor];
}

- (void)buildMesh
{
    // A simple full-screen quad (2 triangles)
    // Position (x,y,z,w), TexCoord (u,v)
    static const Vertex quadVertices[] =
    {
        // Pixel positions, Texture coordinates
        { {  1.0,  -1.0, 0.0, 1.0 },  { 1.0, 1.0 } },
        { { -1.0,  -1.0, 0.0, 1.0 },  { 0.0, 1.0 } },
        { { -1.0,   1.0, 0.0, 1.0 },  { 0.0, 0.0 } },
        
        { {  1.0,  -1.0, 0.0, 1.0 },  { 1.0, 1.0 } },
        { { -1.0,   1.0, 0.0, 1.0 },  { 0.0, 0.0 } },
        { {  1.0,   1.0, 0.0, 1.0 },  { 1.0, 0.0 } },
    };
    
    _vertexBuffer = [_device newBufferWithBytes:quadVertices
                                          length:sizeof(quadVertices)
                                         options:MTLResourceStorageModeShared];
    _numVertices = sizeof(quadVertices) / sizeof(Vertex);
}

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size
{
    _viewportSize.x = size.width;
    _viewportSize.y = size.height;
}

// Chapter 11: Render with animated camera using compute shader
- (void)renderSphereSilhouette
{
    if (!_computePipelineState || !_sphereTexture || !_timeBuffer) {
        NSLog(@"Cannot render: compute pipeline, texture, or time buffer not ready");
        return;
    }
    
    // Update time value in buffer
    float* timePtr = (float*)[_timeBuffer contents];
    *timePtr = _time;
    
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    commandBuffer.label = @"SphereSilhouetteCommand";
    
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_computePipelineState];
    [computeEncoder setTexture:_sphereTexture atIndex:0];
    [computeEncoder setBuffer:_timeBuffer offset:0 atIndex:1];  // Pass time to shader
    
    // Dispatch 1920x1080 threads (one per pixel) - Full HD resolution
    MTLSize threadsPerThreadgroup = MTLSizeMake(16, 16, 1);
    MTLSize threadgroups = MTLSizeMake((1920 + 15) / 16, (1080 + 15) / 16, 1);
    
    [computeEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [computeEncoder endEncoding];
    
    [commandBuffer commit];
    
    // Increment time for next frame (60fps -> increment by 1/60 each frame)
    _time += 0.016f;
}

- (void)drawInMTKView:(nonnull MTKView *)view
{
    // Chapter 11: Re-render the scene every frame with updated camera position
    [self renderSphereSilhouette];
    
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    commandBuffer.label = @"MyCommand";
    
    MTLRenderPassDescriptor *renderPassDescriptor = view.currentRenderPassDescriptor;
    
    if(renderPassDescriptor != nil)
    {
        // Clear to black
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
        
        id<MTLRenderCommandEncoder> renderEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
        
        [renderEncoder setRenderPipelineState:_pipelineState];
        [renderEncoder setVertexBuffer:_vertexBuffer offset:0 atIndex:0];
        [renderEncoder setFragmentTexture:_sphereTexture atIndex:0];
        
        // Draw the quad
        [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle
                          vertexStart:0
                          vertexCount:_numVertices];
        
        [renderEncoder endEncoding];
        
        [commandBuffer presentDrawable:view.currentDrawable];
    }
    
    [commandBuffer commit];
}

@end
