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
}

- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)mtkView
{
    self = [super init];
    if(self)
    {
        _device = mtkView.device;
        _commandQueue = [_device newCommandQueue];
        
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

- (void)drawInMTKView:(nonnull MTKView *)view
{
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    commandBuffer.label = @"MyCommand";
    
    MTLRenderPassDescriptor *renderPassDescriptor = view.currentRenderPassDescriptor;
    
    if(renderPassDescriptor != nil)
    {
        // Simple clear color for now
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
        
        id<MTLRenderCommandEncoder> renderEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
        
        [renderEncoder setRenderPipelineState:_pipelineState];
        [renderEncoder setVertexBuffer:_vertexBuffer offset:0 atIndex:0];
        
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
