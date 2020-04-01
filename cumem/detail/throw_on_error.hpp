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

#include "prologue.hpp"

#include <cstdio>
#include <exception>
#include "terminate.hpp"


CUMEM_NAMESPACE_OPEN_BRACE

namespace CUMEM_DETAIL_NAMESPACE
{
namespace throw_on_error_detail
{


CUMEM_ANNOTATION
inline void print_error_message(cudaError_t e, const char* message) noexcept
{
#if CUMEM_HAS_CUDART
  printf("Error after %s: %s\n", message, cudaGetErrorString(e));
#else
  printf("Error: %s\n", message);
#endif
}


} // end throw_on_error_detail


CUMEM_ANNOTATION
inline void throw_on_error(cudaError_t e, const char* message)
{
  if(e)
  {
#ifndef __CUDA_ARCH__
    std::string what = std::string(message) + std::string(": ") + cudaGetErrorString(e);
    throw std::runtime_error(what);
#else
    CUMEM_DETAIL_NAMESPACE::throw_on_error_detail::print_error_message(e, message);
    CUMEM_DETAIL_NAMESPACE::terminate();
#endif
  }
}


} // end CUMEM_DETAIL_NAMESPACE

CUMEM_NAMESPACE_CLOSE_BRACE

#include "epilogue.hpp"

