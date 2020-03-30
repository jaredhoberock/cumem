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
  cumem::allocator<int> r;

  int* ptr = r.allocate(1);

  int expected = 13;
  *ptr = expected;
  int result = *ptr;

  assert(expected == result);

  r.deallocate(ptr, 1);
}


__host__ __device__
void test_comparison()
{
  cumem::allocator<int> r0, r1;

  // all malloc_resources are the same
  assert(r0 == r1);
  assert(!(r0 != r1));
}


__host__ __device__
void test_copy_construction()
{
  cumem::allocator<int> r;
  cumem::allocator<int> copy = r;

  assert(r == copy);
}


__host__ __device__
void test_converting_copy_construction()
{
  cumem::allocator<int> r;
  cumem::allocator<float> copy = r;

  // silence unused variable warnings
  (void)copy;
}


void test_throw_on_failure()
{
  cumem::allocator<int> r;

  try
  {
    std::size_t n = std::numeric_limits<std::size_t>::max();
    r.allocate(n);
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

