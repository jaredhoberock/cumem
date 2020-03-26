// note that this header file is special and does not use #pragma once

// The REPO_REQUIRES() macro may be used in a function template's parameter list
// to simulate Concepts.
//
// For example, to selectively enable a function template only for integer types,
// we could do something like this:
//
//     template<class Integer,
//              REPO_REQUIRES(std::is_integral<Integer>::value)
//             >
//     Integer plus_one(Integer x)
//     {
//       return x + 1;
//     }
//

#ifndef REPO_REQUIRES

#  define REPO_CONCATENATE_IMPL(x, y) x##y

#  define REPO_CONCATENATE(x, y) REPO_CONCATENATE_IMPL(x, y)

#  define REPO_MAKE_UNIQUE(x) REPO_CONCATENATE(x, __COUNTER__)

#  define REPO_REQUIRES_IMPL(unique_name, ...) bool unique_name = true, typename std::enable_if<(unique_name and __VA_ARGS__)>::type* = nullptr

#  define REPO_REQUIRES(...) REPO_REQUIRES_IMPL(CUDEX_MAKE_UNIQUE(__deduced_true), __VA_ARGS__)

#elif defined(REPO_REQUIRES)

#  ifdef REPO_CONCATENATE_IMPL
#    undef REPO_CONCATENATE_IMPL
#  endif

#  ifdef REPO_CONCATENATE
#    undef REPO_CONCATENATE
#  endif

#  ifdef REPO_MAKE_UNIQUE
#    undef REPO_MAKE_UNIQUE
#  endif

#  ifdef REPO_REQUIRES_IMPL
#    undef REPO_REQUIRES_IMPL
#  endif

#  ifdef REPO_REQUIRES
#    undef REPO_REQUIRES
#  endif

#endif

