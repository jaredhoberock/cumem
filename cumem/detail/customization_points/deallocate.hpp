#pragma once

#include "../prologue.hpp"

#include <utility>
#include "../static_const.hpp"
#include "../type_traits/is_detected.hpp"


CUMEM_NAMESPACE_OPEN_BRACE


namespace CUMEM_DETAIL_NAMESPACE
{


template<class A, class P, class N>
using deallocate_member_function_t = decltype(std::declval<A>().deallocate(std::declval<P>(), std::declval<N>()));

template<class A, class P, class N>
using has_deallocate_member_function = is_detected<deallocate_member_function_t, A, P, N>;


template<class A, class P, class N>
using deallocate_free_function_t = decltype(deallocate(std::declval<A>(), std::declval<P>(), std::declval<N>()));

template<class A, class P, class N>
using has_deallocate_free_function = is_detected<deallocate_free_function_t, A, P, N>;


// this is the type of deallocate
struct deallocate_customization_point
{
  CUMEM_EXEC_CHECK_DISABLE
  template<class A,
           class P,
           class N,
           CUMEM_REQUIRES(has_deallocate_member_function<A&&,P&&,N&&>::value)
          >
  CUMEM_ANNOTATION
  constexpr auto operator()(A&& a, P&& p, N&& n) const ->
    decltype(std::forward<A>(a).deallocate(std::forward<P>(p), std::forward<N>(n)))
  {
    return std::forward<A>(a).deallocate(std::forward<P>(p), std::forward<N>(n));
  }


  CUMEM_EXEC_CHECK_DISABLE
  template<class A,
           class P,
           class N,
           CUMEM_REQUIRES(!has_deallocate_member_function<A&&,P&&,N&&>::value),
           CUMEM_REQUIRES(has_deallocate_free_function<A&&,P&&,N&&>::value)
          >
  CUMEM_ANNOTATION
  constexpr auto operator()(A&& a, P&& p, N&& n) const ->
    decltype(deallocate(std::forward<A>(a), std::forward<P>(p), std::forward<N>(n)))
  {
    return deallocate(std::forward<A>(a), std::forward<P>(p), std::forward<N>(n));
  }
};


namespace
{


// define the deallocate customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& deallocate = CUMEM_DETAIL_NAMESPACE::static_const<CUMEM_DETAIL_NAMESPACE::deallocate_customization_point>::value;
#else
const __device__ CUMEM_DETAIL_NAMESPACE::deallocate_customization_point deallocate;
#endif


} // end anonymous namespace


} // end CUMEM_DETAIL_NAMESPACE


CUMEM_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

