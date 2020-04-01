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

#include "../prologue.hpp"

#include <type_traits>
#include <utility>
#include "../type_traits/is_detected.hpp"
#include "static_const.hpp"


CUMEM_NAMESPACE_OPEN_BRACE


namespace CUMEM_DETAIL_NAMESPACE
{


template<class P, class... Args>
using construct_at_member_function_t = decltype(std::declval<P>().construct_at(std::declval<Args>()...));

template<class P, class... Args>
using has_construct_at_member_function = is_detected<construct_at_member_function_t, P, Args...>;


template<class P, class... Args>
using construct_at_free_function_t = decltype(construct_at(std::declval<P>(), std::declval<Args>()...));

template<class P, class... Args>
using has_construct_at_free_function = is_detected<construct_at_free_function_t, P, Args...>;


// this is the type of construct_at
struct construct_at_customization_point
{
  CUMEM_EXEC_CHECK_DISABLE
  template<class P,
           class... Args,
           CUMEM_REQUIRES(has_construct_at_member_function<P&&,Args&&...>::value)
          >
  CUMEM_ANNOTATION
  auto operator()(P&& p, Args&&... args) const ->
    decltype(std::forward<P>(p).construct_at(std::forward<Args>(args)...))
  {
    return std::forward<P>(p).construct_at(std::forward<Args>(args)...);
  }


  CUMEM_EXEC_CHECK_DISABLE
  template<class P,
           class... Args,
           CUMEM_REQUIRES(!has_construct_at_member_function<P&&,Args&&...>::value),
           CUMEM_REQUIRES(has_construct_at_free_function<P&&,Args&&...>::value)
          >
  CUMEM_ANNOTATION
  auto operator()(P&& p, Args&&... args) const ->
    decltype(construct(std::forward<P>(p), std::forward<Args>(args)...))
  {
    return construct_at(std::forward<P>(p), std::forward<Args>(args)...);
  }


  CUMEM_EXEC_CHECK_DISABLE
  template<class T,
           class... Args,
           CUMEM_REQUIRES(!has_construct_at_member_function<T*,Args&&...>::value),
           CUMEM_REQUIRES(!has_construct_at_free_function<T*,Args&&...>::value),
           CUMEM_REQUIRES(!std::is_void<T>::value)
          >
  CUMEM_ANNOTATION
  void operator()(T* p, Args&&... args) const
  {
    // use placement new
    new(p) T(std::forward<Args>(args)...);
  }
};


namespace
{


// define the construct_at customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& construct_at = CUMEM_DETAIL_NAMESPACE::static_const<CUMEM_DETAIL_NAMESPACE::construct_at_customization_point>::value;
#else
const __device__ CUMEM_DETAIL_NAMESPACE::construct_at_customization_point construct_at;
#endif


} // end anonymous namespace


} // end CUMEM_DETAIL_NAMESPACE


CUMEM_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"


