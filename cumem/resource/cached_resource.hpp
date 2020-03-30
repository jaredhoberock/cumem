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

#include <map>
#include <mutex>
#include <type_traits>


CUMEM_NAMESPACE_OPEN_BRACE


// cached_resource is a memory resource adaptor that maintains a cache of free
// allocations. Calls to allocate draw from this cache first before requesting
// fresh allocations from its base resource. When fresh allocations are
// expensive (as in the case of cudaMalloc), this caching scheme may result in
// improved memory allocation performance.
//
// XXX In principle, cached_resource could be a __host__ __device__ type,
// but we would need to implement its cache using a GPU-friendly key/value store.
template<class MemoryResource>
class cached_resource : private MemoryResource // inherit from MemoryResource for the empty base class optimization
{
  public:
    using base_resource_type = MemoryResource;

    template<class... Args,
             CUMEM_REQUIRES(std::is_constructible<base_resource_type,Args&&...>::value)
            >
    explicit cached_resource(Args&&... args)
      : base_resource_type{std::forward<Args>(args)...}
    {}

    cached_resource() = default;

    cached_resource(const cached_resource&) = delete;

    ~cached_resource()
    {
      deallocate_free_blocks();

      // note that we do not deallocate allocated blocks
      // because that is the responsibility of clients
    }

    base_resource_type& base_resource()
    {
      return *this;
    }

    const base_resource_type& base_resource() const
    {
      return *this;
    }

    void* allocate(size_t num_bytes)
    {
      void* ptr = nullptr;

      // XXX this algorithm searches for a block of exactly the right size, but
      //     in principle we could look for any block that could accomodate the request
      auto free_block = free_blocks_.find(num_bytes);
      if(free_block != free_blocks_.end())
      {
        ptr = free_block->second;

        // erase from the free blocks map
        free_blocks_.erase(free_block);
      }
      else
      {
        // no allocation of the right size exists
        // create a new allocation with the base resource
        ptr = base_resource().allocate(num_bytes);
      }

      // insert the allocation into the allocated_blocks map
      allocated_blocks_.insert(std::make_pair(ptr, num_bytes));

      return ptr;
    }

    void deallocate(void* ptr, size_t)
    {
      // find the allocation in the allocated blocks map
      auto found = allocated_blocks_.find(ptr);

      // insert the block into the free blocks map
      free_blocks_.insert(std::make_pair(found->second, found->first));

      // erase the allocation from the allocated blocks map
      allocated_blocks_.erase(found);
    }

    bool is_equal(const cached_resource& other) const
    {
      return this == &other;
    }

    bool operator==(const cached_resource& other) const
    {
      return is_equal(other);
    }

    bool operator!=(const cached_resource& other) const
    {
      return !(*this == other);
    }

  private:
    // a map from block sizes to free blocks of that size
    std::multimap<std::size_t, void*> free_blocks_;

    // a map from allocated blocks to their sizes
    std::map<void*, std::size_t> allocated_blocks_;

    void deallocate_free_blocks()
    {
      for(auto b : free_blocks_)
      {
        // since this function is only called from the destructor,
        // catch and discard any exceptions we encounter to avoid
        // letting exceptions escape destructors
        try
        {
          base_resource().deallocate(b.second, b.first);
        }
        catch(...)
        {
          // any exception is discarded
        }
      }

      free_blocks_.clear();
    }
};


CUMEM_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

