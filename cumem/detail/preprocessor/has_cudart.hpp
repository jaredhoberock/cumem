// note that this header file is special and does not use #pragma once

// CUMEM_HAS_CUDART indicates whether or not the CUDA Runtime API is available.

#ifndef CUMEM_HAS_CUDART

#  if defined(__CUDACC__)
#    if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__>= 350 && defined(__CUDACC_RDC__))
#      define CUMEM_HAS_CUDART 1
#    else
#      define CUMEM_HAS_CUDART 0
#    endif
#  else
#    define CUMEM_HAS_CUDART 0
#  endif

#elif defined(CUMEM_HAS_CUDART)
#  undef CUMEM_HAS_CUDART
#endif

