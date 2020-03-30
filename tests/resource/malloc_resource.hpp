#include <cassert>
#include <cumem/resource/malloc_resource.hpp>
#include <limits>


#ifndef __CUDACC__
#define __host__
#define __device__
#endif


__host__ __device__
void test_allocate()
{
  cumem::malloc_resource r;

  int* ptr = static_cast<int*>(r.allocate(sizeof(int)));

  int expected = 13;
  *ptr = expected;
  int result = *ptr;

  assert(expected == result);

  r.deallocate(ptr, sizeof(int));
}


__host__ __device__
void test_comparison()
{
  cumem::malloc_resource r0, r1;

  // all malloc_resources are the same
  assert(r0.is_equal(r1));
  assert(r0 == r1);
  assert(!(r0 != r1));
}


__host__ __device__
void test_copy_construction()
{
  cumem::malloc_resource r;
  cumem::malloc_resource copy = r;

  assert(r == copy);
}


void test_throw_on_failure()
{
  cumem::malloc_resource r;

  try
  {
    std::size_t num_bytes = std::numeric_limits<std::size_t>::max();
    r.allocate(num_bytes);
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


void test_malloc_resource()
{
  test_allocate();
  test_comparison();
  test_copy_construction();
  test_throw_on_failure();

#ifdef __CUDACC__
  device_invoke<<<1,1>>>([] __device__ ()
  {
    test_allocate();
    test_comparison();
    test_copy_construction();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

