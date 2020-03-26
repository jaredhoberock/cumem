// note that this header file is special and does not use #pragma once

// REPO_HAS_CUDART indicates whether or not the CUDA Runtime API is available.

#ifndef REPO_HAS_CUDART

#  if defined(__CUDACC__)
#    if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__>= 350 && defined(__CUDACC_RDC__))
#      define REPO_HAS_CUDART 1
#    else
#      define REPO_HAS_CUDART 0
#    endif
#  else
#    define REPO_HAS_CUDART 0
#  endif

#elif defined(REPO_HAS_CUDART)
#  undef REPO_HAS_CUDART
#endif

