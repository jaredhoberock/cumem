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
#include <type_traits>
#include <utility>


CUMEM_NAMESPACE_OPEN_BRACE


// allocator_adaptor adapts a memory resource, which allocates untyped bytes, into
// an allocator, which allocates typed objects.
//
// The name allocator_adaptor was chosen by analogy to N3916's resource_adaptor
// which performs the inverse adaptation from allocator to resource.
template<class T, class MemoryResource>
class allocator_adaptor : private MemoryResource // inherit from MemoryResource for the empty base class optimization
{
  private:
    // XXX alternatively, we could just store a reference to the MemoryResource
    static_assert(std::is_copy_constructible<MemoryResource>::value, "MemoryResource must be copy constructible.");

    using super_t = MemoryResource;

  public:
    using value_type = T;

    CUMEM_EXEC_CHECK_DISABLE
    allocator_adaptor() = default;

    CUMEM_EXEC_CHECK_DISABLE
    allocator_adaptor(const allocator_adaptor&) = default;

    CUMEM_EXEC_CHECK_DISABLE
    template<class U>
    CUMEM_ANNOTATION
    allocator_adaptor(const allocator_adaptor<U,MemoryResource>& other)
      : super_t(other.resource())
    {}

    CUMEM_EXEC_CHECK_DISABLE
    template<class... Args,
             CUMEM_REQUIRES(std::is_constructible<super_t,Args&&...>::value)
            >
    CUMEM_ANNOTATION
    allocator_adaptor(Args&&... args)
      : super_t(std::forward<Args>(args)...)
    {}

    CUMEM_EXEC_CHECK_DISABLE
    template<class U = T,
             CUMEM_REQUIRES(!std::is_void<U>::value)
            >
    CUMEM_ANNOTATION
    value_type* allocate(std::size_t n)
    {
      return reinterpret_cast<value_type*>(resource().allocate(n * sizeof(T)));
    }

    CUMEM_EXEC_CHECK_DISABLE
    template<class U = T,
             CUMEM_REQUIRES(!std::is_void<U>::value)
            >
    CUMEM_ANNOTATION
    void deallocate(value_type* ptr, std::size_t n)
    {
      resource().deallocate(ptr, n);
    }

    CUMEM_ANNOTATION
    const MemoryResource& resource() const
    {
      return *this;
    }

    CUMEM_EXEC_CHECK_DISABLE
    CUMEM_ANNOTATION
    bool operator==(const allocator_adaptor& other) const
    {
      return resource().is_equal(other.resource());
    }

    CUMEM_ANNOTATION
    bool operator!=(const allocator_adaptor& other) const
    {
      return !(*this == other);
    }

  private:
    CUMEM_EXEC_CHECK_DISABLE
    MemoryResource& resource()
    {
      return *this;
    }
};


CUMEM_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

