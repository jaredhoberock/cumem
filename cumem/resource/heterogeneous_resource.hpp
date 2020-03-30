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

#include <cstdint>


CUMEM_NAMESPACE_OPEN_BRACE


// heterogeneous_resource uses a different primitive resource depending on
// whether its operations are called from __host__ or __device__ code.
template<class HostResource, class DeviceResource>
class heterogeneous_resource
{
  public:
    using host_resource_type = HostResource;
    using device_resource_type = DeviceResource;

    CUMEM_EXEC_CHECK_DISABLE
    CUMEM_ANNOTATION
    heterogeneous_resource() :
#ifndef __CUDA_ARCH__
      host_resource_{}
#else
      device_resource_{}
#endif
    {}

    CUMEM_ANNOTATION
    void* allocate(std::size_t num_bytes)
    {
#ifndef __CUDA_ARCH__
      return host_resource_.allocate(num_bytes);
#else
      return device_resource_.allocate(num_bytes);
#endif
    }

    CUMEM_ANNOTATION
    void deallocate(void* ptr, std::size_t num_bytes)
    {
#ifndef __CUDA_ARCH__
      host_resource_.deallocate(ptr, num_bytes);
#else
      device_resource_.deallocate(ptr, num_bytes);
#endif
    }

    CUMEM_ANNOTATION
    bool is_equal(const heterogeneous_resource& other) const
    {
#ifndef __CUDA_ARCH__
      return host_resource_.is_equal(other.host_resource_);
#else
      return device_resource_.is_equal(other.device_resource_);
#endif
    }

    CUMEM_ANNOTATION
    bool operator==(const heterogeneous_resource& other) const
    {
      return is_equal(other);
    }

    CUMEM_ANNOTATION
    bool operator!=(const heterogeneous_resource& other) const
    {
      return !(*this == other);
    }

  private:
#ifndef __CUDA_ARCH__
    host_resource_type host_resource_;
#else
    device_resource_type device_resource_;
#endif
};


CUMEM_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

