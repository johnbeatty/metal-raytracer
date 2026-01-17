#ifndef SharedTypes_h
#define SharedTypes_h

#include <simd/simd.h>

typedef struct {
    vector_float4 components;
} Tuple;

// Constants for types
#define TUPLE_POINT 1.0
#define TUPLE_VECTOR 0.0

typedef struct {
    vector_float4 position;
    vector_float2 textureCoordinate;
} Vertex;

#endif /* SharedTypes_h */
