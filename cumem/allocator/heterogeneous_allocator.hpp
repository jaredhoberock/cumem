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

#include "../resource/heterogeneous_resource.hpp"
#include "../resource/malloc_resource.hpp"
#include "../resource/managed_resource.hpp"
#include "allocator_adaptor.hpp"

CUMEM_NAMESPACE_OPEN_BRACE


// heterogeneous_allocator uses a different primitive resource depending on whether
// its operations are called from __host__ or __device__ code.
template<class T, class HostResource = managed_resource, class DeviceResource = malloc_resource>
class heterogeneous_allocator : public allocator_adaptor<T,heterogeneous_resource<HostResource,DeviceResource>>
{
  private:
    using super_t = allocator_adaptor<T,heterogeneous_resource<HostResource,DeviceResource>>;

  public:
    heterogeneous_allocator() = default;

    heterogeneous_allocator(const heterogeneous_allocator&) = default;

    template<class U>
    CUMEM_ANNOTATION
    heterogeneous_allocator(const heterogeneous_allocator<U,HostResource,DeviceResource>& other)
      : super_t(other)
    {}
}; // end heterogeneous_allocator


CUMEM_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

