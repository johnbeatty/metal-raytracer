#import <MetalKit/MetalKit.h>

@interface Renderer : NSObject <MTKViewDelegate>

- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)mtkView;

// Chapter 5: Render sphere silhouette
- (void)renderSphereSilhouette;

@end
