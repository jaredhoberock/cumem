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

#include <mutex>
#include <type_traits>
#include <utility>


CUMEM_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
class concurrent_value
{
  private:
    std::mutex mutex_;
    T value_;

  public:
    template<class... Args,
             CUMEM_REQUIRES(std::is_constructible<T,Args&&...>::value)
            >
    explicit concurrent_value(Args&&... args)
      : value_(std::forward<Args>(args)...)
    {}

    template<class Function>
    auto exclusive_invoke(Function&& f)
      -> decltype(std::forward<Function>(f)(value_))
    {
      // acquire the mutex associated with the singleton
      std::lock_guard<std::mutex> lock(mutex_);

      // invoke the function on the value while we have the mutex
      return std::forward<Function>(f)(value_);
    }

    template<class Function>
    auto unsafe_invoke(Function&& f) const
      -> decltype(std::forward<Function>(f)(value_))
    {
      return std::forward<Function>(f)(value_);
    }
};


} // end detail


CUMEM_NAMESPACE_CLOSE_BRACE


#include "epilogue.hpp"

