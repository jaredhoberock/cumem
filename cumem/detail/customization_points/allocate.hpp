#pragma once

#include "../prologue.hpp"

#include <utility>
#include "../type_traits/is_detected.hpp"
#include "static_const.hpp"


CUMEM_NAMESPACE_OPEN_BRACE


namespace CUMEM_DETAIL_NAMESPACE
{


template<class A, class N>
using allocate_member_function_t = decltype(std::declval<A>().allocate(std::declval<N>()));

template<class A, class N>
using has_allocate_member_function = is_detected<allocate_member_function_t, A, N>;


template<class A, class N>
using allocate_free_function_t = decltype(allocate(std::declval<A>(), std::declval<N>()));

template<class A, class N>
using has_allocate_free_function = is_detected<allocate_free_function_t, A, N>;


// this is the type of allocate
struct allocate_customization_point
{
  CUMEM_EXEC_CHECK_DISABLE
  template<class A,
           class N,
           CUMEM_REQUIRES(has_allocate_member_function<A&&,N&&>::value)
          >
  CUMEM_ANNOTATION
  constexpr auto operator()(A&& a, N&& n) const ->
    decltype(std::forward<A>(a).allocate(std::forward<N>(n)))
  {
    return std::forward<A>(a).allocate(std::forward<N>(n));
  }


  CUMEM_EXEC_CHECK_DISABLE
  template<class A,
           class N,
           CUMEM_REQUIRES(!has_allocate_member_function<A&&,N&&>::value),
           CUMEM_REQUIRES(has_allocate_free_function<A&&,N&&>::value)
          >
  CUMEM_ANNOTATION
  constexpr auto operator()(A&& a, N&& n) const ->
    decltype(allocate(std::forward<A>(a), std::forward<N>(n)))
  {
    return allocate(std::forward<A>(a), std::forward<N>(n));
  }
};


namespace
{


// define the allocate customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& allocate = CUMEM_DETAIL_NAMESPACE::static_const<CUMEM_DETAIL_NAMESPACE::allocate_customization_point>::value;
#else
const __device__ CUMEM_DETAIL_NAMESPACE::allocate_customization_point allocate;
#endif


} // end anonymous namespace


} // end CUMEM_DETAIL_NAMESPACE


CUMEM_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

