// note that this header file is special and does not use #pragma once

// REPO_HAS_EXCEPTIONS indicates whether or not exception support is available.

#ifndef REPO_HAS_EXCEPTIONS

#  if defined(__CUDACC__)
#    if !defined(__CUDA_ARCH__)
#      define REPO_HAS_EXCEPTIONS __cpp_exceptions
#    else
#      define REPO_HAS_EXCEPTIONS 0
#    endif
#  else
#    define REPO_HAS_EXCEPTIONS __cpp_exceptions
#  endif

#elif defined(REPO_HAS_EXCEPTIONS)
#  undef REPO_HAS_EXCEPTIONS
#endif

