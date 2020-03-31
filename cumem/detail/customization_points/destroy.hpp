#pragma once

#include "../prologue.hpp"

#include <type_traits>
#include <utility>
#include "../type_traits/is_detected.hpp"
#include "static_const.hpp"


CUMEM_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class A, class P>
using destroy_member_function_t = decltype(std::declval<A>().destroy(std::declval<P>()));

template<class A, class P>
using has_destroy_member_function = is_detected<destroy_member_function_t, A, P>;


template<class A, class P>
using destroy_free_function_t = decltype(destroy(std::declval<A>(), std::declval<P>()));

template<class A, class P>
using has_destroy_free_function = is_detected<destroy_free_function_t, A, P>;


// this is the type of destroy
struct destroy_customization_point
{
  CUMEM_EXEC_CHECK_DISABLE
  template<class A,
           class P,
           CUMEM_REQUIRES(has_destroy_member_function<A&&,P&&>::value)
          >
  CUMEM_ANNOTATION
  auto operator()(A&& a, P&& p) const ->
    decltype(std::forward<A>(a).destroy(std::forward<P>(p)))
  {
    return std::forward<A>(a).destroy(std::forward<P>(p));
  }


  CUMEM_EXEC_CHECK_DISABLE
  template<class A,
           class P,
           CUMEM_REQUIRES(!has_destroy_member_function<A&&,P&&>::value),
           CUMEM_REQUIRES(has_destroy_free_function<A&&,P&&>::value)
          >
  CUMEM_ANNOTATION
  auto operator()(A&& a, P&& p) const ->
    decltype(destroy(std::forward<A>(a), std::forward<P>(p)))
  {
    return destroy(std::forward<A>(a), std::forward<P>(p));
  }


  CUMEM_EXEC_CHECK_DISABLE
  template<class A,
           class T,
           CUMEM_REQUIRES(!has_destroy_member_function<A&&,T*>::value),
           CUMEM_REQUIRES(!has_destroy_free_function<A&&,T*>::value),
           CUMEM_REQUIRES(!std::is_void<T>::value)
          >
  CUMEM_ANNOTATION
  void operator()(A&&, T* p) const
  {
    // call the destructor directly
    p->~T();

    // XXX consider delegating this to a hypothetical CPO named destroy_at(p)
  }
};


namespace
{


// define the destroy customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& destroy = detail::static_const<detail::destroy_customization_point>::value;
#else
const __device__ detail::destroy_customization_point destroy;
#endif


} // end anonymous namespace


} // end detail


CUMEM_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

