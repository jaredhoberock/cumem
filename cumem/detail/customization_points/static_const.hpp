#pragma once

#include "../prologue.hpp"

CUMEM_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
struct static_const
{
  static constexpr T value{};
};


// provide the definition of static_const<T>::value
template<class T>
constexpr T static_const<T>::value;


} // end detail


CUMEM_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

