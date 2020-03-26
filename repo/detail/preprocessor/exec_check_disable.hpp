// note that this header file is special and does not use #pragma once

// REPO_EXEC_CHECK_DISABLE expands to a pragma which tells a CUDA-capable
// compiler not to enforce that a function must call another function with
// a compatible execution space

#ifndef REPO_EXEC_CHECK_DISABLE
#  ifdef __CUDACC__
#    define REPO_EXEC_CHECK_DISABLE #pragma nv_exec_check_disable
#  else
#    define REPO_EXEC_CHECK_DISABLE
#  endif
#elif defined(REPO_EXEC_CHECK_DISABLE)
#  undef REPO_EXEC_CHECK_DISABLE
#endif

