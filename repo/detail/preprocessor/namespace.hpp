// note that this header file is special and does not use #pragma once

#if !defined(REPO_NAMESPACE)

// this branch is taken the first time this header is included

#  if defined(REPO_NAMESPACE_OPEN_BRACE) or defined(REPO_NAMESPACE_CLOSE_BRACE)
#    error "Either all of REPO_NAMESPACE, REPO_NAMESPACE_OPEN_BRACE, and REPO_NAMESPACE_CLOSE_BRACE must be defined, or none of them."
#  endif

#  define REPO_NAMESPACE cudex
#  define REPO_NAMESPACE_OPEN_BRACE namespace cudex {
#  define REPO_NAMESPACE_CLOSE_BRACE }
#  define REPO_NAMESPACE_NEEDS_UNDEF

#elif defined(REPO_NAMESPACE_NEEDS_UNDEF)

// this branch is taken the second time this header is included

#  undef REPO_NAMESPACE
#  undef REPO_NAMESPACE_OPEN_BRACE
#  undef REPO_NAMESPACE_CLOSE_BRACE
#  undef REPO_NAMESPACE_NEEDS_UNDEF

#elif defined(REPO_NAMESPACE) or defined(REPO_NAMESPACE_OPEN_BRACE) or defined(REPO_CLOSE_BRACE)

// this branch is taken the first time this header is included, and the user has misconfigured these namespace-related symbols

#  if !defined(REPO_NAMESPACE) or !defined(REPO_NAMESPACE_OPEN_BRACE) or !defined(REPO_NAMESPACE_CLOSE_BRACE)
#    error "Either all of REPO_NAMESPACE, REPO_NAMESPACE_OPEN_BRACE, and REPO_NAMESPACE_CLOSE_BRACE must be defined, or none of them."
#  endif

#endif

