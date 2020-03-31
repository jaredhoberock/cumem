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

#include "../resource/system_resource.hpp"
#include "../resource/managed_resource.hpp"
#include "heterogeneous_allocator.hpp"


CUMEM_NAMESPACE_OPEN_BRACE


// allocator<T> is the default allocator type. It is an alias for the following recipe:
//
//     heterogeneous_allocator<T, system_resource<managed_resource>>>
//
// That means that when on the host, allocator allocates CUDA managed memory through a system-wide memory pool.
// When on the device, allocator allocates CUDA __device__ memory through malloc.
template<class T>
class allocator : public heterogeneous_allocator<T, system_resource<managed_resource>>
{
  private:
    using super_t = heterogeneous_allocator<T, system_resource<managed_resource>>;

  public:
    allocator() = default;

    CUMEM_ANNOTATION
    explicit allocator(int device)
      : super_t{device}
    {}

    allocator(const allocator&) = default;

    template<class U>
    CUMEM_ANNOTATION
    allocator(const allocator<U>& other)
      : super_t(other)
    {}
};


CUMEM_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

