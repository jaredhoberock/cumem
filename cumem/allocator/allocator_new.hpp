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
#include "../detail/customization_points/allocate.hpp"
#include "../detail/customization_points/construct.hpp"

CUMEM_NAMESPACE_OPEN_BRACE


template<class T, class Alloc>
using allocator_new_t = typename std::allocator_traits<Alloc>::template rebind_traits<T>::pointer;


CUMEM_EXEC_CHECK_DISABLE
template<class T, class Alloc, class... Args>
CUMEM_ANNOTATION
allocator_new_t<T,Alloc> allocator_new(Alloc& alloc, Args&&... args)
{
  using allocator_type = typename std::allocator_traits<Alloc>::template rebind_alloc<T>;
  allocator_type alloc_copy = alloc;

  auto p = detail::allocate(alloc_copy, 1);
  detail::construct(alloc_copy, p, std::forward<Args>(args)...);

  return p;
}


CUMEM_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

