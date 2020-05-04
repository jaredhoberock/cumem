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

#include <memory>
#include <type_traits>
#include "../detail/swap.hpp"
#include "allocator_delete.hpp"


CUMEM_NAMESPACE_OPEN_BRACE


// see https://wg21.link/P0211
// allocation_deleter is a deleter which uses an Allocator to delete objects
template<class Allocator>
class allocation_deleter : private Allocator // use inheritance for empty base class optimization
{
  public:
    using pointer = typename std::allocator_traits<Allocator>::pointer;

    CUMEM_ANNOTATION
    explicit allocation_deleter(const Allocator& alloc) noexcept
      : Allocator(alloc)
    {}

    template<class OtherAllocator,
             CUMEM_REQUIRES(std::is_convertible<typename std::allocator_traits<OtherAllocator>::pointer, pointer>::value)
            >
    CUMEM_ANNOTATION
    allocation_deleter(const allocation_deleter<OtherAllocator>& other) noexcept
      : Allocator(other.allocator())
    {}

    CUMEM_EXEC_CHECK_DISABLE
    CUMEM_ANNOTATION
    void operator()(pointer ptr)
    {
      CUMEM_NAMESPACE::allocator_delete(as_allocator(), ptr);
    }

    CUMEM_ANNOTATION
    Allocator& allocator()
    {
      return *this;
    }

    CUMEM_ANNOTATION
    const Allocator& allocator() const
    {
      return *this;
    }

    CUMEM_ANNOTATION
    void swap(allocation_deleter& other)
    {
      CUMEM_DETAIL_NAMESPACE::swap(as_allocator(), other.as_allocator());
    }

  private:
    CUMEM_ANNOTATION
    Allocator& as_allocator()
    {
      return const_cast<allocation_deleter&>(*this);
    }

    // allocation_deleter's copy constructor needs access to mutable allocation_deleter<OtherAllocator>::as_allocator
    template<class OtherAllocator> friend class allocation_deleter;
};


CUMEM_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

