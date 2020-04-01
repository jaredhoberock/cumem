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
#include <cuda_runtime.h>
#include "../detail/invoke_with_current_device.hpp"


CUMEM_NAMESPACE_OPEN_BRACE


class managed_resource
{
  public:
    // note that this does not check that the device actually exists
    inline explicit managed_resource(int device)
      : device_(device)
    {}

    inline managed_resource()
      : managed_resource{0}
    {}

    managed_resource(const managed_resource&) = default;

    inline void* allocate(std::size_t num_bytes) const
    {
      return CUMEM_DETAIL_NAMESPACE::invoke_with_current_device(device(), [=]
      {
        void* result = nullptr;

        CUMEM_DETAIL_NAMESPACE::throw_on_error(cudaMallocManaged(&result, num_bytes, cudaMemAttachGlobal), "managed_resource::allocate: CUDA error after cudaMallocManaged");

        return result;
      });
    }

    inline void deallocate(void* ptr, std::size_t) const
    {
      CUMEM_DETAIL_NAMESPACE::invoke_with_current_device(device(), [=]
      {
        CUMEM_DETAIL_NAMESPACE::throw_on_error(cudaFree(ptr), "managed_resource::deallocate: CUDA error after cudaFree");
      });
    }

    inline int device() const
    {
      return device_;
    }

    inline bool is_equal(const managed_resource& other) const
    {
      return device() == other.device();
    }

    inline bool operator==(const managed_resource& other) const
    {
      return is_equal(other);
    }

    inline bool operator!=(const managed_resource& other) const
    {
      return !(*this == other);
    }

  private:
    int device_;
};

CUMEM_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

