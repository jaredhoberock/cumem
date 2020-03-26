// note that this header file is special and does not use #pragma once

// CUMEM_HAS_EXCEPTIONS indicates whether or not exception support is available.

#ifndef CUMEM_HAS_EXCEPTIONS

#  if defined(__CUDACC__)
#    if !defined(__CUDA_ARCH__)
#      define CUMEM_HAS_EXCEPTIONS __cpp_exceptions
#    else
#      define CUMEM_HAS_EXCEPTIONS 0
#    endif
#  else
#    define CUMEM_HAS_EXCEPTIONS __cpp_exceptions
#  endif

#elif defined(CUMEM_HAS_EXCEPTIONS)
#  undef CUMEM_HAS_EXCEPTIONS
#endif

