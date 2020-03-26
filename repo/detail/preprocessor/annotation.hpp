// note that this header file is special and does not use #pragma once

// REPO_ANNOTATION expands to __host__ __device__ when encountered by a
// CUDA-capable compiler

#if !defined(REPO_ANNOTATION)

#  ifdef __CUDACC__
#    define REPO_ANNOTATION __host__ __device__
#  else
#    define REPO_ANNOTATION
#  endif
#  define REPO_ANNOTATION_NEEDS_UNDEF

#elif defined(REPO_ANNOTATION_NEEDS_UNDEF)

#undef REPO_ANNOTATION
#undef REPO_ANNOTATION_NEEDS_UNDEF

#endif

