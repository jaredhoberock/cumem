// note that this header file is special and does not use #pragma once

// CUMEM_ANNOTATION expands to __host__ __device__ when encountered by a
// CUDA-capable compiler

#if !defined(CUMEM_ANNOTATION)

#  ifdef __CUDACC__
#    define CUMEM_ANNOTATION __host__ __device__
#  else
#    define CUMEM_ANNOTATION
#  endif
#  define CUMEM_ANNOTATION_NEEDS_UNDEF

#elif defined(CUMEM_ANNOTATION_NEEDS_UNDEF)

#undef CUMEM_ANNOTATION
#undef CUMEM_ANNOTATION_NEEDS_UNDEF

#endif

