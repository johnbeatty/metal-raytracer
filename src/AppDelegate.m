#import "AppDelegate.h"

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    NSRect frame = NSMakeRect(0, 0, 800, 600);
    
    self.window = [[NSWindow alloc] initWithContentRect:frame
                                              styleMask:(NSWindowStyleMaskTitled |
                                                         NSWindowStyleMaskClosable |
                                                         NSWindowStyleMaskResizable |
                                                         NSWindowStyleMaskMiniaturizable)
                                                backing:NSBackingStoreBuffered
                                                  defer:NO];
    [self.window setTitle:@"Ray Tracer Challenge"];
    [self.window center];
    
    // Create MTKView
    self.view = [[MTKView alloc] initWithFrame:frame];
    self.view.device = MTLCreateSystemDefaultDevice();
    
    if (!self.view.device) {
        NSLog(@"Metal is not supported on this device");
        return;
    }
    
    self.renderer = [[Renderer alloc] initWithMetalKitView:self.view];
    
    if (!self.renderer) {
        NSLog(@"Renderer failed initialization");
        return;
    }
    
    [self.renderer mtkView:self.view drawableSizeWillChange:self.view.drawableSize];
    
    self.view.delegate = self.renderer;
    self.window.contentView = self.view;
    [self.window makeKeyAndOrderFront:nil];
    self.window.delegate = self;
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}

@end
