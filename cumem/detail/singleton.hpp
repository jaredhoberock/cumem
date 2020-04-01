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

#include <map>
#include <type_traits>
#include <utility>
#include "concurrent_value.hpp"


CUMEM_NAMESPACE_OPEN_BRACE


namespace CUMEM_DETAIL_NAMESPACE
{


// This is a container for at most one value.
// The existence of the value is tracked by an external variable.
// When the container is destroyed, the variable is set to indicate
// the object's lifetime has ended.
template<class T>
class intrusive_optional
{
  public:
    template<class... Args,
             CUMEM_REQUIRES(std::is_constructible<T,Args&&...>::value)
            >
    intrusive_optional(bool& has_value, Args&&... args)
      : value_{std::forward<Args>(args)...},
        has_value_(has_value)
    {
      has_value_ = true;
    }

    ~intrusive_optional()
    {
      has_value_ = false;
    }

    T& value()
    {
      return value_;
    }

  private:
    T value_;
    bool& has_value_;
};


template<class T>
inline T* singleton()
{
  // singleton<T>() may be called after static destructors have completed.
  // In this case, the resource's lifetime has ended. Track its lifetime with a variable
  // and store it in an intrusive_optional.
  //
  // These conditions are why we return a pointer instead of a reference.
  // Returning a pointer allows us to return nullptr when the resource's lifetime has ended.

  // define a variable to track the lifetime of resource
  static bool has_value = false;

  // pass the variable to the intrusive_optional's ctor
  static intrusive_optional<T> resource(has_value);

  // make sure the resource actually exists before returning it 
  return has_value ? &resource.value() : nullptr;
}


// this is an interface to a associative map from a Key to
// a singleton T constructed from that Key
template<class T, class Key,
         CUMEM_REQUIRES(std::is_constructible<T,Key>::value)
        >
T& find_concurrent_singleton(const Key& key)
{
  using map_type = concurrent_value<std::map<Key,T>>;

  // get the singleton map
  map_type& singleton_map = *CUMEM_DETAIL_NAMESPACE::singleton<map_type>();

  // find the singleton object of interest, or create it if it
  // doesn't exist
  return singleton_map.exclusive_invoke([=](std::map<Key,T>& map) -> T&
  {
    auto found = map.find(key);
    if(found == map.end())
    {
      // create the object
      found = map.emplace(std::piecewise_construct,
                          std::make_tuple(key),    // ctor args for the key
                          std::make_tuple(key)     // ctor args for the value
      ).first;
    }

    return found->second;
  });
}


// this is a view of a singleton T constructed and associated with a Key
template<class Key, class T>
class concurrent_singleton_view
{
  private:
    static_assert(std::is_constructible<T,Key>::value, "T must be constructible from Key.");

    concurrent_value<T>* singleton_;

  public:
    explicit concurrent_singleton_view(const Key& key)
      : singleton_(&find_concurrent_singleton<concurrent_value<T>>(key))
    {}

    concurrent_singleton_view(const concurrent_singleton_view&) = default;

    template<class Function>
    auto exclusive_invoke(Function&& f)
      -> decltype(singleton_->exclusive_invoke(std::forward<Function>(f)))
    {
      return singleton_->exclusive_invoke(std::forward<Function>(f));
    }

    template<class Function>
    auto unsafe_invoke(Function&& f) const
      -> decltype(singleton_->unsafe_invoke(std::forward<Function>(f)))
    {
      return singleton_->unsafe_invoke(std::forward<Function>(f));
    }
};
  

} // end CUMEM_DETAIL_NAMESPACE


CUMEM_NAMESPACE_CLOSE_BRACE


#include "epilogue.hpp"

