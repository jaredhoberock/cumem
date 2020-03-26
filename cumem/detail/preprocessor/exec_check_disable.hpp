// note that this header file is special and does not use #pragma once

// CUMEM_EXEC_CHECK_DISABLE expands to a pragma which tells a CUDA-capable
// compiler not to enforce that a function must call another function with
// a compatible execution space

#ifndef CUMEM_EXEC_CHECK_DISABLE
#  ifdef __CUDACC__
#    define CUMEM_EXEC_CHECK_DISABLE #pragma nv_exec_check_disable
#  else
#    define CUMEM_EXEC_CHECK_DISABLE
#  endif
#elif defined(CUMEM_EXEC_CHECK_DISABLE)
#  undef CUMEM_EXEC_CHECK_DISABLE
#endif

