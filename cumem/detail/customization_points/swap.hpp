#pragma once

#include "../prologue.hpp"

#include <type_traits>
#include <utility>
#include "../type_traits/is_detected.hpp"
#include "static_const.hpp"


CUMEM_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
using swap_member_function_t = decltype(std::declval<T>().swap(std::declval<T>()));

template<class T>
using has_swap_member_function = is_detected<swap_member_function_t, T>;


template<class T>
using swap_free_function_t = decltype(swap(std::declval<T>(), std::declval<T>()));

template<class T>
using has_swap_free_function = is_detected<swap_free_function_t, T>;


// this is the type of swap
struct swap_customization_point
{
  CUMEM_EXEC_CHECK_DISABLE
  template<class T,
           CUMEM_REQUIRES(has_swap_member_function<T&>::value)
          >
  CUMEM_ANNOTATION
  constexpr auto operator()(T& a, T& b) const ->
    decltype(a.swap(b))
  {
    return a.swap(b);
  }


  CUMEM_EXEC_CHECK_DISABLE
  template<class T,
           CUMEM_REQUIRES(!has_swap_member_function<T&>::value),
           CUMEM_REQUIRES(has_swap_free_function<T&>::value)
          >
  CUMEM_ANNOTATION
  constexpr auto operator()(T& a, T& b) const ->
    decltype(swap(a, b))
  {
    return swap(a, b);
  }


  CUMEM_EXEC_CHECK_DISABLE
  template<class T,
           CUMEM_REQUIRES(!has_swap_member_function<T&>::value),
           CUMEM_REQUIRES(!has_swap_free_function<T&>::value),
           CUMEM_REQUIRES(std::is_move_constructible<T>::value)
          >
  CUMEM_ANNOTATION
  void operator()(T& a, T& b) const
  {
    T old_a{std::move(a)};
    a = std::move(b);
    b = std::move(old_a);
  }
};


namespace
{


// define the swap customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& swap = detail::static_const<detail::swap_customization_point>::value;
#else
const __device__ detail::swap_customization_point swap;
#endif


} // end anonymous namespace


} // end detail


CUMEM_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

