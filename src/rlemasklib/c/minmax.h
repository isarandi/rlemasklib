#pragma once
#include "basics.h"

#define RLEMASKLIB_MAX(a, b) ((a) > (b) ? (a) : (b))
#define RLEMASKLIB_MIN(a, b) ((a) < (b) ? (a) : (b))
#define RLEMASKLIB_CLIP(val, low, high) ((val) <= (low) ? (low) : ((val) > (high) ? (high) : (val)))

#define RLEMASKLIB_MAKE_MIN(type) \
    static type type##Min(type a, type b) { \
        return RLEMASKLIB_MIN(a, b); \
    }

#define RLEMASKLIB_MAKE_MAX(type) \
    static type type##Max(type a, type b) { \
        return RLEMASKLIB_MAX(a, b); \
    }

#define RLEMASKLIB_MAKE_CLIP(type) \
    static type type##Clip(type x, type min_, type max_) { \
        return RLEMASKLIB_CLIP(x, min_, max_); \
    }

#define RLEMASKLIB_MAKE_ALL(type) \
    RLEMASKLIB_MAKE_MIN(type) \
    RLEMASKLIB_MAKE_MAX(type) \
    RLEMASKLIB_MAKE_CLIP(type)

RLEMASKLIB_MAKE_ALL(int)
RLEMASKLIB_MAKE_ALL(uint)
RLEMASKLIB_MAKE_ALL(siz)
RLEMASKLIB_MAKE_ALL(double)

#undef RLEMASKLIB_MAKE_MIN
#undef RLEMASKLIB_MAKE_MAX
#undef RLEMASKLIB_MAKE_CLIP
#undef RLEMASKLIB_MAKE_ALL

