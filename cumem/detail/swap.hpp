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

#include "prologue.hpp"

#include <type_traits>
#include <utility>
#include "static_const.hpp"
#include "type_traits/is_detected.hpp"


CUMEM_NAMESPACE_OPEN_BRACE


namespace CUMEM_DETAIL_NAMESPACE
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
constexpr auto const& swap = CUMEM_DETAIL_NAMESPACE::static_const<CUMEM_DETAIL_NAMESPACE::swap_customization_point>::value;
#else
const __device__ CUMEM_DETAIL_NAMESPACE::swap_customization_point swap;
#endif


} // end anonymous namespace


} // end CUMEM_DETAIL_NAMESPACE


CUMEM_NAMESPACE_CLOSE_BRACE

#include "epilogue.hpp"

