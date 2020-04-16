#pragma once

#include "prologue.hpp"

CUMEM_NAMESPACE_OPEN_BRACE


namespace CUMEM_DETAIL_NAMESPACE
{


template<class T>
struct static_const
{
  static constexpr T value{};
};


// provide the definition of static_const<T>::value
template<class T>
constexpr T static_const<T>::value;


} // end CUMEM_DETAIL_NAMESPACE


CUMEM_NAMESPACE_CLOSE_BRACE

#include "epilogue.hpp"

