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
#include "construct_at.hpp"


CUMEM_NAMESPACE_OPEN_BRACE


namespace CUMEM_DETAIL_NAMESPACE
{


template<class A, class P, class... Args>
using construct_member_function_t = decltype(std::declval<A>().construct(std::declval<P>(), std::declval<Args>()...));

template<class A, class P, class... Args>
using has_construct_member_function = is_detected<construct_member_function_t, A, P, Args...>;


template<class A, class P, class... Args>
using construct_free_function_t = decltype(construct(std::declval<A>(), std::declval<P>(), std::declval<Args>()...));

template<class A, class P, class... Args>
using has_construct_free_function = is_detected<construct_free_function_t, A, P, Args...>;


// this is the type of construct
struct construct_customization_point
{
  CUMEM_EXEC_CHECK_DISABLE
  template<class A,
           class P,
           class... Args,
           CUMEM_REQUIRES(has_construct_member_function<A&&,P&&,Args&&...>::value)
          >
  CUMEM_ANNOTATION
  auto operator()(A&& a, P&& p, Args&&... args) const ->
    decltype(std::forward<A>(a).construct(std::forward<P>(p), std::forward<Args>(args)...))
  {
    return std::forward<A>(a).construct(std::forward<P>(p), std::forward<Args>(args)...);
  }


  CUMEM_EXEC_CHECK_DISABLE
  template<class A,
           class P,
           class... Args,
           CUMEM_REQUIRES(!has_construct_member_function<A&&,P&&,Args&&...>::value),
           CUMEM_REQUIRES(has_construct_free_function<A&&,P&&,Args&&...>::value)
          >
  CUMEM_ANNOTATION
  auto operator()(A&& a, P&& p, Args&&... args) const ->
    decltype(construct(std::forward<A>(a), std::forward<P>(p), std::forward<Args>(args)...))
  {
    return construct(std::forward<A>(a), std::forward<P>(p), std::forward<Args>(args)...);
  }


  CUMEM_EXEC_CHECK_DISABLE
  template<class A,
           class T,
           class... Args,
           CUMEM_REQUIRES(!has_construct_member_function<A&&,T*,Args&&...>::value),
           CUMEM_REQUIRES(!has_construct_free_function<A&&,T*,Args&&...>::value),
           CUMEM_REQUIRES(!std::is_void<T>::value)
          >
  CUMEM_ANNOTATION
  void operator()(A&&, T* p, Args&&... args) const
  {
    CUMEM_NAMESPACE::construct_at(p, std::forward<Args>(args)...);
  }
};


} // end CUMEM_DETAIL_NAMESPACE


namespace
{


// define the construct customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& construct = CUMEM_DETAIL_NAMESPACE::static_const<CUMEM_DETAIL_NAMESPACE::construct_customization_point>::value;
#else
const __device__ CUMEM_DETAIL_NAMESPACE::construct_customization_point construct;
#endif


} // end anonymous namespace


CUMEM_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

