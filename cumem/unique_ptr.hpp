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

#include <memory>
#include <type_traits>
#include <utility>
#include "allocator/allocation_deleter.hpp"
#include "allocator/allocator.hpp"
#include "allocator/construct.hpp"
#include "detail/static_const.hpp"
#include "detail/swap.hpp"
#include "detail/type_traits/is_detected.hpp"


CUMEM_NAMESPACE_OPEN_BRACE


namespace CUMEM_DETAIL_NAMESPACE
{


template<class T>
struct nested_pointer
{
  using type = typename T::pointer;
};

template<class T>
using nested_pointer_t = typename nested_pointer<T>::type;


} // end CUMEM_DETAIL_NAMESPACE


template<class T>
class default_delete : public allocation_deleter<allocator<T>>
{
  private:
    using super_t = allocation_deleter<allocator<T>>;

  public:
    using super_t::super_t;

    CUMEM_ANNOTATION
    default_delete()
      : super_t(allocator<T>())
    {}
};


template<class T, class Deleter = default_delete<T>>
class unique_ptr
{
  public:
    using element_type = typename std::decay<T>::type;

    using pointer = CUMEM_DETAIL_NAMESPACE::detected_or_t<
      T*,
      CUMEM_DETAIL_NAMESPACE::nested_pointer_t,
      typename std::remove_reference<Deleter>::type
    >;

    using deleter_type = Deleter;

    CUMEM_ANNOTATION
    unique_ptr(pointer ptr, const deleter_type& deleter)
      : ptr_(ptr),
        deleter_(deleter)
    {}

    CUMEM_ANNOTATION
    unique_ptr(pointer ptr) noexcept
      : unique_ptr(ptr, deleter_type{})
    {}

    CUMEM_ANNOTATION
    unique_ptr() noexcept : unique_ptr(nullptr) {}

    CUMEM_ANNOTATION
    unique_ptr(unique_ptr&& other) noexcept
      : ptr_(),
        deleter_(std::move(other.get_deleter()))
    {
      CUMEM_DETAIL_NAMESPACE::swap(ptr_, other.ptr_);
      CUMEM_DETAIL_NAMESPACE::swap(deleter_, other.deleter_);
    }

    template<class OtherT,
             class OtherDeleter,
             CUMEM_REQUIRES(std::is_convertible<typename unique_ptr<OtherT,OtherDeleter>::pointer, pointer>::value),
             CUMEM_REQUIRES(std::is_constructible<Deleter, OtherDeleter&&>::value)
            >
    CUMEM_ANNOTATION
    unique_ptr(unique_ptr<OtherT,OtherDeleter>&& other) noexcept
      : ptr_(other.release()),
        deleter_(std::move(other.get_deleter()))
    {}
  
    CUMEM_ANNOTATION
    ~unique_ptr()
    {
      reset(nullptr);
    }

    CUMEM_ANNOTATION
    unique_ptr& operator=(unique_ptr&& other) noexcept
    {
      CUMEM_DETAIL_NAMESPACE::swap(ptr_, other.ptr_);
      CUMEM_DETAIL_NAMESPACE::swap(deleter_, other.deleter_);
      return *this;
    }

    CUMEM_ANNOTATION
    pointer get() const noexcept
    {
      return ptr_;
    }

    CUMEM_ANNOTATION
    pointer release() noexcept
    {
      pointer result = nullptr;
      CUMEM_DETAIL_NAMESPACE::swap(ptr_, result);
      return result;
    }

    CUMEM_ANNOTATION
    void reset(pointer ptr = pointer()) noexcept
    {
      CUMEM_DETAIL_NAMESPACE::swap(ptr_, ptr);

      if(ptr != nullptr)
      {
        get_deleter()(ptr); 
      }
    }

    CUMEM_ANNOTATION
    deleter_type& get_deleter() noexcept
    {
      return deleter_;
    }

    CUMEM_ANNOTATION
    const deleter_type& get_deleter() const noexcept
    {
      return deleter_;
    }

    CUMEM_ANNOTATION
    typename std::add_lvalue_reference<T>::type operator*() const
    {
      return *ptr_;
    }

    CUMEM_ANNOTATION
    operator bool () const noexcept
    {
      return get();
    }

    CUMEM_ANNOTATION
    void swap(unique_ptr& other) noexcept
    {
      CUMEM_DETAIL_NAMESPACE::swap(ptr_, other.ptr_);
      CUMEM_DETAIL_NAMESPACE::swap(deleter_, other.deleter_);
    }

  private:
    T* ptr_;
    deleter_type deleter_;
};


struct uninitialized_t{};


namespace
{


#ifndef __CUDA_ARCH__
constexpr uninitialized_t const& uninitialized = CUMEM_DETAIL_NAMESPACE::static_const<uninitialized_t>::value;
#else
const __device__ uninitialized_t uninitialized;
#endif


} // end anonymous namespace


CUMEM_EXEC_CHECK_DISABLE
template<class T, class Alloc, class Deleter>
CUMEM_ANNOTATION
unique_ptr<T,Deleter> allocate_unique_with_deleter(const Alloc& alloc, const Deleter& deleter, uninitialized_t)
{
  using allocator_type = typename std::allocator_traits<Alloc>::template rebind_alloc<T>;
  allocator_type alloc_copy = alloc;
  Deleter deleter_copy = deleter;

  unique_ptr<T,Deleter> result(alloc_copy.allocate(1), deleter_copy);

  return std::move(result);
}


CUMEM_EXEC_CHECK_DISABLE
template<class T, class Alloc, class Deleter, class... Args>
CUMEM_ANNOTATION
unique_ptr<T,Deleter> allocate_unique_with_deleter(const Alloc& alloc, const Deleter& deleter, Args&&... args)
{
  unique_ptr<T,Deleter> result = CUMEM_NAMESPACE::allocate_unique_with_deleter<T>(alloc, deleter, uninitialized);

  CUMEM_NAMESPACE::construct(alloc, result.get(), std::forward<Args>(args)...);

  return std::move(result);
}


CUMEM_EXEC_CHECK_DISABLE
template<class T, class Alloc, class... Args>
CUMEM_ANNOTATION
unique_ptr<T,allocation_deleter<Alloc>> allocate_unique(const Alloc& alloc, Args&&... args)
{
  // because allocation_deleter<Alloc> derives T from its Alloc parameter,
  // we need to rebind Alloc to T before giving it to deleter
  using allocator_type = typename std::allocator_traits<Alloc>::template rebind_alloc<T>;
  allocator_type alloc_copy = alloc;
  allocation_deleter<allocator_type> deleter{alloc_copy};

  return CUMEM_NAMESPACE::allocate_unique_with_deleter<T>(alloc, deleter, std::forward<Args>(args)...);
}


template<class T, class... Args>
CUMEM_ANNOTATION
unique_ptr<T> make_unique(Args&&... args)
{
  return CUMEM_NAMESPACE::allocate_unique_with_deleter<T>(allocator<T>(), default_delete<T>{}, std::forward<Args>(args)...);
}


CUMEM_NAMESPACE_CLOSE_BRACE

#include "detail/epilogue.hpp"

