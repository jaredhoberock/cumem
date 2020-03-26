// note that this header file is special and does not use #pragma once

// The CUMEM_REQUIRES() macro may be used in a function template's parameter list
// to simulate Concepts.
//
// For example, to selectively enable a function template only for integer types,
// we could do something like this:
//
//     template<class Integer,
//              CUMEM_REQUIRES(std::is_integral<Integer>::value)
//             >
//     Integer plus_one(Integer x)
//     {
//       return x + 1;
//     }
//

#ifndef CUMEM_REQUIRES

#  define CUMEM_CONCATENATE_IMPL(x, y) x##y

#  define CUMEM_CONCATENATE(x, y) CUMEM_CONCATENATE_IMPL(x, y)

#  define CUMEM_MAKE_UNIQUE(x) CUMEM_CONCATENATE(x, __COUNTER__)

#  define CUMEM_REQUIRES_IMPL(unique_name, ...) bool unique_name = true, typename std::enable_if<(unique_name and __VA_ARGS__)>::type* = nullptr

#  define CUMEM_REQUIRES(...) CUMEM_REQUIRES_IMPL(CUMEM_MAKE_UNIQUE(__deduced_true), __VA_ARGS__)

#elif defined(CUMEM_REQUIRES)

#  ifdef CUMEM_CONCATENATE_IMPL
#    undef CUMEM_CONCATENATE_IMPL
#  endif

#  ifdef CUMEM_CONCATENATE
#    undef CUMEM_CONCATENATE
#  endif

#  ifdef CUMEM_MAKE_UNIQUE
#    undef CUMEM_MAKE_UNIQUE
#  endif

#  ifdef CUMEM_REQUIRES_IMPL
#    undef CUMEM_REQUIRES_IMPL
#  endif

#  ifdef CUMEM_REQUIRES
#    undef CUMEM_REQUIRES
#  endif

#endif

