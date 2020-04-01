// note that this header file is special and does not use #pragma once

#if !defined(CUMEM_NAMESPACE)

// this branch is taken the first time this header is included

#  if defined(CUMEM_NAMESPACE_OPEN_BRACE) or defined(CUMEM_NAMESPACE_CLOSE_BRACE)
#    error "Either all of CUMEM_NAMESPACE, CUMEM_NAMESPACE_OPEN_BRACE, and CUMEM_NAMESPACE_CLOSE_BRACE must be defined, or none of them."
#  endif

#  define CUMEM_NAMESPACE cumem
#  define CUMEM_NAMESPACE_OPEN_BRACE namespace cumem {
#  define CUMEM_NAMESPACE_CLOSE_BRACE }
#  define CUMEM_NAMESPACE_NEEDS_UNDEF

#elif defined(CUMEM_NAMESPACE_NEEDS_UNDEF)

// this branch is taken the second time this header is included

#  undef CUMEM_NAMESPACE
#  undef CUMEM_NAMESPACE_OPEN_BRACE
#  undef CUMEM_NAMESPACE_CLOSE_BRACE
#  undef CUMEM_NAMESPACE_NEEDS_UNDEF

#elif defined(CUMEM_NAMESPACE) or defined(CUMEM_NAMESPACE_OPEN_BRACE) or defined(CUMEM_CLOSE_BRACE)

// this branch is taken the first time this header is included, and the user has misconfigured these namespace-related symbols

#  if !defined(CUMEM_NAMESPACE) or !defined(CUMEM_NAMESPACE_OPEN_BRACE) or !defined(CUMEM_NAMESPACE_CLOSE_BRACE)
#    error "Either all of CUMEM_NAMESPACE, CUMEM_NAMESPACE_OPEN_BRACE, and CUMEM_NAMESPACE_CLOSE_BRACE must be defined, or none of them."
#  endif

#endif

#if !defined(CUMEM_DETAIL_NAMESPACE)

// allow the user to define a singly-nested namespace for private implementation details
#  define CUMEM_DETAIL_NAMESPACE detail
#  define CUMEM_DETAIL_NAMESPACE_NEEDS_UNDEF

#elif defined(CUMEM_DETAIL_NAMESPACE_NEEDS_UNDEF)

#  undef CUMEM_DETAIL_NAMESPACE
#  undef CUMEM_DETAIL_NAMESPACE_NEEDS_UNDEF

#endif

