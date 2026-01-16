#ifndef SharedTypes_h
#define SharedTypes_h

#include <simd/simd.h>

typedef struct {
    vector_float4 components;
} Tuple;

// Constants for types
#define TUPLE_POINT 1.0
#define TUPLE_VECTOR 0.0

#endif /* SharedTypes_h */
