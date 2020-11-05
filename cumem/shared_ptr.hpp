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

#include "detail/prologue.hpp"

#include <type_traits>
#include "allocator/allocator.hpp"
#include "allocator/allocator_delete.hpp"
#include "allocator/allocator_new.hpp"
#include "detail/exchange.hpp"
#include "detail/swap.hpp"
#include "unique_ptr.hpp"


CUMEM_NAMESPACE_OPEN_BRACE


template<class T, class Deleter = default_delete<T>>
class shared_ptr
{
  private:
    static_assert(std::is_copy_constructible<Deleter>::value, "shared_ptr Deleter type must be copy constructible.");

    using unique_ptr_type = unique_ptr<T,Deleter>;

    struct control_block
    {
      unique_ptr_type ptr_to_object;
      int use_count;

      CUMEM_ANNOTATION
      control_block(unique_ptr_type&& ptr, int count)
        : ptr_to_object{std::move(ptr)},
          use_count{count}
      {}
    };

    // XXX consider type-erasing the allocator
    allocator<control_block> alloc_;
    control_block* ptr_to_control_block_;

    CUMEM_ANNOTATION
    long increment_use_count()
    {
#ifndef __CUDA_ARCH__
      return __atomic_fetch_add(&ptr_to_control_block_->use_count, 1, __ATOMIC_SEQ_CST);
#elif (__CUDA_ARCH_ < 600)
      return atomicAdd(&ptr_to_control_block_->use_count, 1);
#else
      return atomicAdd_system(&ptr_to_control_block_->use_count, 1);
#endif
    }

    CUMEM_ANNOTATION
    long decrement_use_count()
    {
#ifndef __CUDA_ARCH__
      return __atomic_fetch_add(&ptr_to_control_block_->use_count, -1, __ATOMIC_SEQ_CST);
#elif (__CUDA_ARCH_ < 600)
      return atomicAdd(&ptr_to_control_block_->use_count, -1);
#else
      return atomicAdd_system(&ptr_to_control_block_->use_count, -1);
#endif
    }

  public:
    using element_type = typename unique_ptr_type::element_type;
    using deleter_type = Deleter;
    using pointer = typename deleter_type::pointer;

    CUMEM_ANNOTATION
    shared_ptr(pointer ptr, const deleter_type& deleter)
      : alloc_{},
        ptr_to_control_block_{ptr != nullptr ? allocator_new<control_block>(alloc_, unique_ptr_type{ptr, deleter}, 1) : nullptr}
    {}

    CUMEM_ANNOTATION
    shared_ptr(pointer ptr)
      : shared_ptr{ptr, deleter_type{}}
    {}

    CUMEM_ANNOTATION
    constexpr shared_ptr(std::nullptr_t) noexcept
      : alloc_{},
        ptr_to_control_block_{nullptr}
    {}

    CUMEM_ANNOTATION
    constexpr shared_ptr()
      : shared_ptr{nullptr}
    {}

    CUMEM_ANNOTATION
    shared_ptr(shared_ptr&& other)
      : alloc_{std::move(other.alloc_)},
        ptr_to_control_block_{CUMEM_DETAIL_NAMESPACE::exchange(other.ptr_to_control_block_, nullptr)}
    {}

    CUMEM_ANNOTATION
    shared_ptr(const shared_ptr& other)
      : alloc_{other.alloc_},
        ptr_to_control_block_{other.ptr_to_control_block_}
    {
      increment_use_count();
    }

    CUMEM_ANNOTATION
    ~shared_ptr()
    {
      if(*this and decrement_use_count() == 1)
      {
        // destroy the control block
        allocator_delete(alloc_, ptr_to_control_block_);
      }
    }

    CUMEM_ANNOTATION
    shared_ptr& operator=(shared_ptr& other) noexcept
    {
      shared_ptr{other}.swap(*this);
      return *this;
    }

    CUMEM_ANNOTATION
    shared_ptr& operator=(shared_ptr&& other) noexcept
    {
      alloc_ = std::move(other.alloc_);
      ptr_to_control_block_ = CUMEM_DETAIL_NAMESPACE::exchange(other.ptr_to_control_block_, nullptr);
      return *this;
    }

    CUMEM_ANNOTATION
    typename unique_ptr_type::pointer get() const noexcept
    {
      return ptr_to_control_block_ ? ptr_to_control_block_->ptr_to_object.get() : nullptr;
    }

    CUMEM_ANNOTATION
    typename std::add_lvalue_reference<T>::type operator*() const
    {
      return *ptr_to_control_block_->ptr_to_object;
    }

    CUMEM_ANNOTATION
    operator bool () const noexcept
    {
      return get();
    }

    CUMEM_ANNOTATION
    void swap(shared_ptr& other) noexcept
    {
      CUMEM_DETAIL_NAMESPACE::swap(alloc_, other.alloc_);
      CUMEM_DETAIL_NAMESPACE::swap(ptr_to_control_block_, other.ptr_to_control_block_);
    }

    CUMEM_ANNOTATION
    void reset() noexcept
    {
      shared_ptr{}.swap(*this);
    }

    CUMEM_ANNOTATION
    long use_count() const noexcept
    {
      return ptr_to_control_block_ ? ptr_to_control_block_->use_count : 0;
    }

    CUMEM_ANNOTATION
    bool unique() const noexcept
    {
      return use_count() == 1;
    }

    CUMEM_ANNOTATION
    unique_ptr<T,Deleter> as_unique_ptr() &&
    {
      if(!unique())
      {
        CUMEM_DETAIL_NAMESPACE::throw_runtime_error("shared_ptr::as_unique: not unique");
      }

      unique_ptr<T,Deleter> result = std::move(ptr_to_control_block_->ptr_to_object);

      // leave self in an empty state
      reset();

      return result;
    }

    CUMEM_ANNOTATION
    friend bool operator==(const shared_ptr& a, const shared_ptr& b) noexcept
    {
      return a.get() == b.get();
    }

    CUMEM_ANNOTATION
    friend bool operator!=(const shared_ptr& a, const shared_ptr& b) noexcept
    {
      return !(a == b);
    }
};


template<class T, class Alloc, class... Args,
         CUMEM_REQUIRES(std::is_constructible<T,Args&&...>::value)
        >
CUMEM_ANNOTATION
shared_ptr<T,allocation_deleter<Alloc>> allocate_shared(const Alloc& alloc, Args&&... args)
{
  Alloc alloc_copy = alloc;
  return {allocator_new<T>(alloc_copy, std::forward<Args>(args)...), allocation_deleter<Alloc>{alloc}};
}


template<class T, class... Args,
         CUMEM_REQUIRES(std::is_constructible<T,Args&&...>::value)
        >
CUMEM_ANNOTATION
shared_ptr<T,default_delete<T>> allocate_shared(const allocator<T>& alloc, Args&&... args)
{
  allocator<T> alloc_copy = alloc;
  return {allocator_new<T>(alloc_copy, std::forward<Args>(args)...), default_delete<T>{alloc}};
}


template<class T, class... Args,
         CUMEM_REQUIRES(std::is_constructible<T,Args&&...>::value)
        >
CUMEM_ANNOTATION
shared_ptr<T,default_delete<T>> make_shared(Args&&... args)
{
  return CUMEM_NAMESPACE::allocate_shared<T>(allocator<T>{}, std::forward<Args>(args)...);
}


CUMEM_NAMESPACE_CLOSE_BRACE

