#include <cassert>
#include <cumem/allocator/allocate.hpp>
#include <cumem/allocator/allocator.hpp>
#include <limits>


#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif


__host__ __device__
void test()
{
  cumem::allocator<int> a;

  int* ptr = cumem::allocate(a, 1);

  int expected = 13;
  *ptr = expected;
  int result = *ptr;

  assert(expected == result);

  a.deallocate(ptr, 1);
}


void test_allocate()
{
  test();

#ifdef __CUDACC__
  device_invoke<<<1,1>>>([] __device__ ()
  {
    test();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

