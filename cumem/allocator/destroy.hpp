// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "../detail/prologue.hpp"

#include <type_traits>
#include <utility>
#include "../detail/static_const.hpp"
#include "../detail/type_traits/is_detected.hpp"
#include "destroy_at.hpp"


CUMEM_NAMESPACE_OPEN_BRACE


namespace CUMEM_DETAIL_NAMESPACE
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
    CUMEM_NAMESPACE::destroy_at(p);
  }
};


} // end CUMEM_DETAIL_NAMESPACE


namespace
{


// define the destroy customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& destroy = CUMEM_DETAIL_NAMESPACE::static_const<CUMEM_DETAIL_NAMESPACE::destroy_customization_point>::value;
#else
const __device__ CUMEM_DETAIL_NAMESPACE::destroy_customization_point destroy;
#endif


} // end anonymous namespace


CUMEM_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

