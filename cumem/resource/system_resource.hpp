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

#include <mutex>
#include "../detail/singleton.hpp"
#include "cached_resource.hpp"


CUMEM_NAMESPACE_OPEN_BRACE


// system_resource is a memory resource which allocates from a system-wide singleton memory resource
// on the device indicated by the ctor argument
template<class MemoryResource>
class system_resource
{
  public:
    static_assert(std::is_constructible<MemoryResource,int>::value, "MemoryResource must be constructible from a device ordinal.");

    explicit system_resource(int device)
      : singleton_{device}
    {}

    system_resource()
      : system_resource(0)
    {}

    system_resource(const system_resource&) = default;

    int device() const
    {
      return singleton_.unsafe_invoke([](const singleton_resource_type& self)
      {
        return self.base_resource().device();
      });
    }

    void* allocate(std::size_t num_bytes)
    {
      return singleton_.exclusive_invoke([=](singleton_resource_type& self)
      {
        return self.allocate(num_bytes);
      });
    }

    void deallocate(void* ptr, std::size_t num_bytes)
    {
      singleton_.exclusive_invoke([=](singleton_resource_type& resource)
      {
        resource.deallocate(ptr, num_bytes);
      });
    }

    inline bool is_equal(const system_resource& other) const
    {
      return singleton_.unsafe_invoke([&](const singleton_resource_type& self)
      {
        return other.singleton_.unsafe_invoke([&](const singleton_resource_type& other)
        {
          return self.is_equal(other);
        });
      });
    }

    inline bool operator==(const system_resource& other) const
    {
      return is_equal(other);
    }

    inline bool operator!=(const system_resource& other) const
    {
      return !(*this == other);
    }

  private:
    // since this is intended to be a system-wide resource, cache its allocations
    using singleton_resource_type = cached_resource<MemoryResource>;

    CUMEM_DETAIL_NAMESPACE::concurrent_singleton_view<int, singleton_resource_type> singleton_;
};


CUMEM_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

