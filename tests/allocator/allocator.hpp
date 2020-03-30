#include <cassert>
#include <cumem/allocator/allocator.hpp>
#include <limits>


#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif


__host__ __device__
void test_allocate()
{
  cumem::allocator<int> a;

  int* ptr = a.allocate(1);

  int expected = 13;
  *ptr = expected;
  int result = *ptr;

  assert(expected == result);

  a.deallocate(ptr, 1);
}


__host__ __device__
void test_comparison()
{
  cumem::allocator<int> a0, a1;

  // all malloc_resources are the same
  assert(a0 == a1);
  assert(!(a0 != a1));
}


__host__ __device__
void test_copy_construction()
{
  cumem::allocator<int> a;
  cumem::allocator<int> copy = a;

  assert(a == copy);
}


__host__ __device__
void test_converting_copy_construction()
{
  cumem::allocator<int> a;
  cumem::allocator<float> copy = a;

  // silence unused variable warnings
  (void)copy;
}


void test_throw_on_failure()
{
  cumem::allocator<int> a;

  try
  {
    std::size_t n = std::numeric_limits<std::size_t>::max();
    a.allocate(n);
  }
  catch(...)
  {
    return;
  }

  assert(0);
}


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif


void test_allocator()
{
  test_allocate();
  test_comparison();
  test_copy_construction();
  test_converting_copy_construction();
  test_throw_on_failure();

#ifdef __CUDACC__
  device_invoke<<<1,1>>>([] __device__ ()
  {
    test_allocate();
    test_comparison();
    test_copy_construction();
    test_converting_copy_construction();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

